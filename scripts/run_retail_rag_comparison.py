"""Run the frozen Retail Full/Tool-RAG comparison with a persistent API ledger."""

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path

from aftersalesagent.business.database import clone_database
from aftersalesagent.business.service import RetailService
from aftersalesagent.harness.journal import Journal
from aftersalesagent.knowledge.retail import RetailPolicyIndex
from aftersalesagent.local_config import load_local_config, options_from_settings
from aftersalesagent.models.deepseek import DeepSeekBackend
from aftersalesagent.policies.application import load_retail_core_policy
from aftersalesagent.policies.retail_capabilities import (
    RETAIL_APP_CAPABILITIES,
    RETAIL_RAG_INSTRUCTION,
)
from aftersalesagent.runtime.langgraph_runtime import LangGraphRuntime
from aftersalesagent.schemas.contracts import ConfirmationEvent, MessageEvent, RunContext, RunLimits
from aftersalesagent.tools.registry import ToolRegistry

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/results/retail_agent_rag_v1"
LEDGER = BASE / "api_ledger.json"
CASES = BASE / "scenarios_v1.json"
OUT = BASE / "comparison_v1"
HARDENED = False
WRITES = {
    "R01": "cancel_pending_order",
    "R02": "modify_pending_order_address",
    "R13": "return_delivered_order_items",
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", "utf-8")
    tmp.replace(path)


def read_json(path):
    return json.loads(path.read_text("utf-8"))


class AccountedBackend:
    def __init__(self, backend, run_id):
        self.backend, self.run_id = backend, run_id

    async def generate(self, request):
        ledger = read_json(LEDGER)
        run = ledger["runs"][-1]
        if run["id"] != self.run_id:
            raise RuntimeError("Ledger run mismatch")
        if (
            ledger["completed_api_attempts"] + ledger["reserved_api_attempts"] >= 720
            or run["actual_attempts"] >= run["max_attempts"]
        ):
            raise RuntimeError("API attempt ceiling reached")
        run["actual_attempts"] += 1
        ledger["reserved_api_attempts"] += 1
        write_json(LEDGER, ledger)
        try:
            response = await self.backend.generate(request)
            ledger = read_json(LEDGER)
            run = ledger["runs"][-1]
            run["input_tokens"] += response.input_tokens or 0
            run["output_tokens"] += response.output_tokens or 0
            if response.input_tokens is None or response.output_tokens is None:
                run["unknown_usage_calls"] += 1
            write_json(LEDGER, ledger)
            return response
        finally:
            ledger = read_json(LEDGER)
            ledger["reserved_api_attempts"] -= 1
            ledger["completed_api_attempts"] += 1
            write_json(LEDGER, ledger)


def evidence_for_full(index):
    return {
        c.chunk_id: {
            "chunk_id": c.chunk_id,
            "text": c.text,
            "version": c.version,
            "scope": c.scope,
            "origin": c.origin,
            "source": c.source,
            "section": c.section,
            "rule_ids": index.coverage.get(c.chunk_id, {}).get("rule_ids", [c.chunk_id]),
            "supplied_by": "full_context",
        }
        for c in index.chunks
    }


def policy_for_arm(arm, evidence):
    core = load_retail_core_policy()
    if arm == "tool_rag":
        return core + RETAIL_RAG_INSTRUCTION + RETAIL_APP_CAPABILITIES if HARDENED else core
    core = core.replace(
        "Before asserting a merchant policy condition, refund channel, timeline, or capability "
        "boundary, call search_retail_policy. Cite every such conclusion using [cite:chunk_id] "
        "from the evidence actually returned.",
        "Before asserting a merchant policy condition, refund channel, timeline, or capability "
        "boundary, use the complete policy evidence below. Cite every such conclusion using "
        "[cite:chunk_id] from these evidence cards.",
    )
    if "call search_retail_policy" in core:
        raise ValueError("Full prompt still instructs retrieval")
    cards = "\n\n".join(
        f"[cite:{cid}] {record['section']} ({record['origin']})\n{record['text']}"
        for cid, record in evidence.items()
    )
    return core + "\n\nComplete Retail policy evidence (retail_sim_v1):\n" + cards


def order_snapshot(store, order_id):
    with store.transaction() as connection:
        return store.order(connection, order_id)


def evaluate(case, before, after, first, action, events):
    expected_tool = WRITES.get(case["id"])
    state_ok = after["status"] == case["expected_order_status"]
    if expected_tool:
        action_ok = action is not None and action["tool_name"] == expected_tool
        unchanged_before_confirmation = action is None or before == action["before_order"]
        if case["id"] == "R02":
            address_ok = "10 Pine Street" in json.dumps(after.get("address"), ensure_ascii=False)
        else:
            address_ok = True
        write_ok = action_ok and unchanged_before_confirmation and state_ok and address_ok
    else:
        action_ok = action is None
        write_ok = before == after and action_ok
    tool_names = [e.payload.get("name") for e in events if e.kind == "tool_call"]
    search_count = tool_names.count("search_retail_policy")
    return {
        "expected_write_tool": expected_tool,
        "status_correct": state_ok,
        "write_boundary_correct": write_ok,
        "action_correct": action_ok,
        "citation_count": len(first.sources),
        "search_count": search_count,
        "has_policy_citation": bool(first.sources),
        "termination_reason": first.termination_reason,
        "first_status": first.status,
        "requires_explanation_review": True,
    }


async def run_case(case, arm, options, secret, index, *, limits=None):
    run_id = case["id"] + "-" + arm
    path = OUT / (run_id + ".sqlite")
    if path.exists():
        raise RuntimeError("Run database already exists; no implicit rerun")
    store = clone_database(ROOT / ".local/databases/retail_template.sqlite", path)
    journal = Journal(OUT / (run_id + "-journal.sqlite"))
    service = RetailService(store)
    backend = DeepSeekBackend(options, secret, allow_network=True)
    evidence = evidence_for_full(index) if arm == "full" else {}
    runtime = LangGraphRuntime(
        AccountedBackend(backend, run_id),
        service,
        policy_text=policy_for_arm(arm, evidence),
        registry=ToolRegistry(service, retail_index=index, include_retail_search=arm == "tool_rag"),
        journal=journal,
        trace_directory=OUT / "traces",
    )
    started = time.monotonic()
    try:
        before = order_snapshot(store, case["order_id"])
        view = await runtime.start(
            RunContext(
                environment_id=service.environment_id,
                backend="deepseek",
                model_name=options.model,
                knowledge_version="retail_sim_v1",
                knowledge_fingerprint=index.fingerprint,
                max_output_tokens=1024,
                limits=limits
                or RunLimits(
                    max_model_calls=12,
                    max_tool_calls=20,
                    max_active_seconds=120,
                    max_tokens=48000,
                    max_turn_model_calls=8,
                    max_turn_tool_calls=12,
                    max_turn_active_seconds=90,
                    max_turn_tokens=24000,
                ),
            )
        )
        if evidence:
            frame, _ = await runtime._snapshot(view.session_id)
            frame.evidence = evidence
            await runtime._save(frame)
        first = await runtime.advance(
            view.session_id,
            MessageEvent(event_id=run_id + "-message", content=case["user_message"]),
        )
        action = None
        final = first
        if first.status == "waiting_confirmation":
            pending = first.session.pending_action
            action = pending.model_dump(mode="json")
            action["before_order"] = order_snapshot(store, case["order_id"])
            if case["id"] in WRITES and pending.tool_name == WRITES[case["id"]]:
                final = await runtime.advance(
                    view.session_id,
                    ConfirmationEvent(
                        event_id=run_id + "-confirmation",
                        action_id=pending.action_id,
                        approved=True,
                    ),
                )
        after = order_snapshot(store, case["order_id"])
        events = journal.events(view.session_id)
        assessment = evaluate(case, before, after, first, action, events)
        usage = journal.usage(view.session_id)
        return {
            "id": case["id"],
            "split": case["split"],
            "arm": arm,
            "case": case,
            "session_id": view.session_id,
            "before_order": before,
            "after_order": after,
            "first_status": first.status,
            "first_reply": first.reply,
            "first_sources": first.sources,
            "action": action,
            "final_status": final.status,
            "final_reply": final.reply,
            "usage": usage,
            "elapsed_seconds": time.monotonic() - started,
            "tool_calls": [e.payload for e in events if e.kind == "tool_call"],
            "tool_results": [e.payload for e in events if e.kind == "tool_result"],
            "assessment": assessment,
        }
    finally:
        await backend.aclose()
        journal.close()
        store.close()


async def main():
    global OUT, HARDENED
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--enable-paid-api", action="store_true")
    parser.add_argument("--case", action="append", help="Run a frozen case ID; default all")
    parser.add_argument("--split", choices=["dev", "test"], help="Run one frozen split")
    parser.add_argument("--arm", choices=["full", "tool_rag"], action="append")
    parser.add_argument("--hardened-diagnostic", action="store_true")
    args = parser.parse_args()
    if not args.enable_paid_api:
        parser.error("Explicit paid API flag required")
    if args.hardened_diagnostic:
        if args.split != "dev" or args.arm != ["tool_rag"] or not args.case:
            parser.error("Hardened diagnostic requires dev cases and --arm tool_rag")
        if len(args.case) > 3:
            parser.error("At most three post-test dev diagnostic cases")
        OUT = BASE / "post_test_dev_diagnostic_v1"
        HARDENED = True
    settings, secret = load_local_config(args.env_file)
    options = options_from_settings(settings)
    if not secret.get_secret_value():
        parser.error("Local DeepSeek key missing")
    frozen = read_json(CASES)
    expected_digest = "22afe2e688428206cece421e582365b6eee785a97e8b56d460a3f4ec9e1fca18"
    actual_digest = hashlib.sha256(
        json.dumps(
            {key: value for key, value in frozen.items() if key != "sha256"},
            sort_keys=True,
            ensure_ascii=False,
        ).encode()
    ).hexdigest()
    if frozen["sha256"] != expected_digest or actual_digest != expected_digest:
        raise RuntimeError("Frozen scenario contents changed")
    selected = [
        c
        for c in frozen["cases"]
        if (args.case is None or c["id"] in args.case)
        and (args.split is None or c["split"] == args.split)
    ]
    if args.case and set(args.case) != {c["id"] for c in selected}:
        parser.error("Unknown frozen case")
    arms = args.arm or ["full", "tool_rag"]
    index = RetailPolicyIndex(method="bm25")
    try:
        protocol = {
            "scenario_sha256": frozen["sha256"],
            "index_fingerprint": index.fingerprint,
            "model": options.model,
            "temperature": options.temperature,
            "arm_order": ["full", "tool_rag"],
            "retrieval": "bm25",
            "policy_full_sha256": hashlib.sha256(
                policy_for_arm("full", evidence_for_full(index)).encode()
            ).hexdigest(),
            "policy_tool_sha256": hashlib.sha256(
                policy_for_arm("tool_rag", {}).encode()
            ).hexdigest(),
            "limits": "12 model calls/session; 8/turn; 48000 reported tokens/session; 24000/turn",
        }
        if HARDENED:
            protocol["diagnostic"] = (
                "Post-test, dev-only application prompt guard; excluded from frozen comparison"
            )
        protocol_path = OUT / "protocol.json"
        if protocol_path.exists() and read_json(protocol_path) != protocol:
            raise RuntimeError("Frozen comparison protocol changed")
        write_json(protocol_path, protocol)
        for case in selected:
            for arm in arms:
                run_id = case["id"] + "-" + arm
                result_path = OUT / (run_id + ".json")
                if result_path.exists():
                    continue
                ledger = read_json(LEDGER)
                if ledger["actual_trajectories"] >= ledger["max_trajectories"]:
                    raise RuntimeError("Trajectory ceiling reached")
                ledger["actual_trajectories"] += 1
                ledger["runs"].append(
                    {
                        "kind": "Retail post-test dev diagnostic"
                        if HARDENED
                        else "Retail frozen comparison",
                        "id": run_id,
                        "status": "running",
                        "max_attempts": 12,
                        "actual_attempts": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "unknown_usage_calls": 0,
                    }
                )
                write_json(LEDGER, ledger)
                try:
                    result = await run_case(case, arm, options, secret, index)
                    status = "completed"
                except Exception as exc:
                    status = "failed"
                    result = {"id": case["id"], "arm": arm, "error": type(exc).__name__}
                write_json(result_path, result)
                ledger = read_json(LEDGER)
                ledger["runs"][-1]["status"] = status
                write_json(LEDGER, ledger)
                print(
                    json.dumps(
                        {
                            "run": run_id,
                            "status": status,
                            "first": result.get("first_status"),
                            "final": result.get("final_status"),
                            "state_ok": result.get("assessment", {}).get("status_correct"),
                            "write_ok": result.get("assessment", {}).get("write_boundary_correct"),
                            "error": result.get("error"),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                if status == "failed":
                    return
    finally:
        index.close()


if __name__ == "__main__":
    asyncio.run(main())
