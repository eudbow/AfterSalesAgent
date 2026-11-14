"""固定任务驱动与B0/B1对照；用户脚本不读取私有判分答案。"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from decimal import Decimal
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from aftersalesagent.business.database import clone_database
from aftersalesagent.business.service import RetailService
from aftersalesagent.evaluation.pricing import PriceSchedule, estimate_cost
from aftersalesagent.evaluation.suite import TOOL, Case, public_data
from aftersalesagent.evaluation.verifier import observe, verify
from aftersalesagent.harness.journal import Journal
from aftersalesagent.models.base import ModelBackend
from aftersalesagent.runtime.langgraph_runtime import LangGraphRuntime
from aftersalesagent.schemas.contracts import ConfirmationEvent, MessageEvent, RunContext


@asynccontextmanager
async def runtime_scope(
    store, folder: Path, baseline: str, factory: Callable[[], ModelBackend], policy: str
):
    journal = Journal(folder / "application.sqlite" if baseline == "B1" else None)
    backend = factory()
    try:
        if baseline == "B1":
            async with AsyncSqliteSaver.from_conn_string(
                str(folder / "checkpoints.sqlite")
            ) as saver:
                yield LangGraphRuntime(
                    backend,
                    RetailService(store),
                    policy_text=policy,
                    journal=journal,
                    checkpointer=saver,
                )
        else:
            yield LangGraphRuntime(
                backend, RetailService(store), policy_text=policy, journal=journal
            )
    finally:
        closer = getattr(backend, "aclose", None)
        if closer:
            await closer()
        journal.close()


def matches_request(task, action) -> bool:
    data = public_data(task)
    return (
        data["intent"] in {"cancel", "address", "return"}
        and action.tool_name == TOOL[data["intent"]]
        and action.arguments == data["arguments"]
        and action.order_id == data["order_id"]
    )


async def run_case(
    case: Case,
    *,
    template: Path,
    folder: Path,
    baseline: str,
    backend_factory: Callable[[], ModelBackend],
    policy: str,
    context: RunContext,
    profile: str = "normal",
    pricing: PriceSchedule | None = None,
) -> dict:
    folder.mkdir(parents=True, exist_ok=False)
    store = clone_database(template, folder / "business.sqlite")
    failures = []
    started = time.monotonic()
    try:
        async with runtime_scope(store, folder, baseline, backend_factory, policy) as runtime:
            session = await runtime.start(
                context.model_copy(update={"environment_id": store.path.stem})
            )
            sid = session.session_id
            initial = observe(store, sid)
            result = await runtime.advance(
                sid,
                MessageEvent(
                    event_id="message-1",
                    content=case.public.first_user_message,
                ),
            )
            # 脚本只重复客户已给出的信息，不从criteria补答案或动态改变任务。
            for index in range(2):
                if (
                    result.status not in {"completed", "waiting_user"}
                    or case.public_intent == "query"
                ):
                    break
                if result.session.pending_action is not None:
                    break
                if case.public_intent == "cancel" and any(
                    word in result.reply.lower()
                    for word in ("cannot", "can't", "无法", "不能", "not allowed")
                ):
                    break
                result = await runtime.advance(
                    sid,
                    MessageEvent(
                        event_id=f"clarification-{index + 1}",
                        content="Please use the customer details in my original REQUEST_DATA. "
                        "If the request is disallowed, explain it. Otherwise ask for confirmation.",
                    ),
                )
            action = result.session.pending_action
            approved = bool(
                action and matches_request(case.public, action) and case.approve_matching
            )
            if action and not matches_request(case.public, action):
                failures.append("unexpected_proposal_rejected_by_user")
            restart = profile == "restart_before_confirm" and action is not None
            if profile == "commit_response_loss" and action is not None:
                original = runtime.service.confirm
                attempts = []

                def lose_once(session_id: str, action_id: str, approved: bool):
                    value = original(session_id, action_id, approved)
                    attempts.append(1)
                    if len(attempts) == 1:
                        raise TimeoutError("Injected response loss after commit")
                    return value

                runtime.service.confirm = lose_once
            if action is not None and not restart:
                result = await runtime.advance(
                    sid,
                    ConfirmationEvent(
                        event_id="confirm-1",
                        action_id=action.action_id,
                        approved=approved,
                    ),
                )
            events = runtime.journal.events(sid)
            usage = runtime.journal.usage(sid)
            final = observe(store, sid)
        if restart:
            async with runtime_scope(store, folder, baseline, backend_factory, policy) as restored:
                try:
                    await restored.get_state(sid)
                    assert action is not None
                    result = await restored.advance(
                        sid,
                        ConfirmationEvent(
                            event_id="confirm-1",
                            action_id=action.action_id,
                            approved=approved,
                        ),
                    )
                    events, usage = restored.journal.events(sid), restored.journal.usage(sid)
                    final = observe(store, sid)
                except ValueError:
                    failures.append("conversation_unavailable_after_restart")
        verdict = verify(
            case.criteria, initial, final, events, result.reply, result.status, failures
        )
        trace = folder / "trace.jsonl"
        trace.write_text("".join(e.model_dump_json() + "\n" for e in events), encoding="utf-8")
        (folder / "final_state.json").write_text(
            json.dumps(final, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        blocked = sum(
            e.kind == "tool_result"
            and isinstance(error := e.payload.get("error"), dict)
            and error.get("code") in {"policy_denied", "invalid_state"}
            for e in events
        )
        report = {
            "task_id": case.public.task_id,
            "split": case.split,
            "family": case.family,
            "kind": case.criteria.kind,
            "baseline": baseline,
            "profile": profile,
            "verification": verdict.model_dump(mode="json"),
            "terminal_status": result.status,
            "termination_reason": result.termination_reason,
            "reply": result.reply,
            "blocked_attempts": blocked,
            "usage": usage,
            "wall_seconds": time.monotonic() - started,
            "trace": str(trace),
            "driver_failures": failures,
            "fault_applied": profile != "normal" and action is not None,
            "cost_estimate": estimate_cost(events, pricing),
            "cost_note": "Unknown usage or pricing means unknown cost, not zero",
        }
        (folder / "result.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return report
    finally:
        store.close()


def summarize(rows: list[dict], *, backend: str, manifest: dict, config: dict) -> dict:
    groups = {}
    for baseline in sorted({row["baseline"] for row in rows}):
        selected = [row for row in rows if row["baseline"] == baseline]
        n = len(selected)
        passed = sum(row["verification"]["success"] for row in selected)
        failures = {}
        for row in selected:
            for label in row["verification"]["failure_labels"]:
                failures[label] = failures.get(label, 0) + 1
        safety = sum(
            not row["verification"]["checks"]["no_unauthorized_actual_changes"] for row in selected
        )
        duplicate = sum(
            not row["verification"]["checks"]["no_duplicate_actual_execution"] for row in selected
        )
        affected = [row for row in selected if row["fault_applied"]]
        recovered = sum(row["verification"]["success"] for row in affected)
        estimates = [row["cost_estimate"] for row in selected]
        total_cost = (
            {
                "amount": str(sum(Decimal(item["amount"]) for item in estimates)),
                "currency": estimates[0]["currency"],
                "estimated_not_invoice": True,
            }
            if estimates and all(item is not None for item in estimates)
            else None
        )
        groups[baseline] = {
            "cases": n,
            "passed": passed,
            "task_success_rate": passed / n if backend == "deepseek" else None,
            "plumbing_pass_rate": passed / n if backend == "replay" else None,
            "failure_labels": failures,
            "actual_safety_violations": safety,
            "actual_violation_rate": safety / n,
            "duplicate_executions": duplicate,
            "duplicate_task_rate": duplicate / n,
            "blocked_attempts": sum(row["blocked_attempts"] for row in selected),
            "model_calls": sum(row["usage"]["model_calls"] for row in selected),
            "tool_calls": sum(row["usage"]["tool_calls"] for row in selected),
            "reported_tokens": sum(row["usage"]["reported_tokens"] for row in selected),
            "unknown_usage_calls": sum(row["usage"]["unknown_usage_calls"] for row in selected),
            "cost_estimate": total_cost,
            "fault_exposed_cases": len(affected),
            "recovered_cases": recovered,
            "scripted_recovery_rate": recovered / len(affected) if affected else None,
            "model_api_failures": sum(
                str(row["termination_reason"]).startswith("model_")
                and row["terminal_status"] == "failed"
                for row in selected
            ),
            "budget_truncations": sum(row["terminal_status"] == "truncated" for row in selected),
        }
    return {
        "mode": "live_adapted_task_evaluation"
        if backend == "deepseek"
        else "offline_public_request_replay",
        "suite_id": manifest["suite_id"],
        "suite_file_hashes": manifest["files"],
        "config": config,
        "baselines": groups,
        "official_tau2_score": False,
        "limitations": [
            "Structured project-authored tasks, not official tau2 benchmark",
            "Reply evidence is a coarse deterministic check; human audit remains necessary",
            "Replay validates plumbing only; it does not measure an LLM",
            "Safety/version check failure counts require trace audit, not assumed incident rates",
        ],
    }
