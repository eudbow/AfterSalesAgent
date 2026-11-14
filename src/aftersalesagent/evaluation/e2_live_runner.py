"""E2 Native/MCP真实模型验收；驱动器不替模型修参数，失败不重跑覆盖。"""

import json
import re
import time
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from aftersalesagent.business.database import clone_database
from aftersalesagent.business.service import RetailService
from aftersalesagent.evaluation.e1_runner import verify as verify_knowledge
from aftersalesagent.evaluation.nl_driver import ScriptedUser
from aftersalesagent.evaluation.nl_replay import ScriptedNLReplay
from aftersalesagent.evaluation.nl_runner import reply_evidence
from aftersalesagent.evaluation.nl_suite import UserPlan
from aftersalesagent.evaluation.suite import CaseCriteria
from aftersalesagent.evaluation.verifier import observe, verify
from aftersalesagent.harness.journal import Journal
from aftersalesagent.knowledge.factory import make_service
from aftersalesagent.knowledge.service import KnowledgeQuestion
from aftersalesagent.mcp_transport.adapters import MCPRetailProvider
from aftersalesagent.runtime.langgraph_runtime import LangGraphRuntime
from aftersalesagent.schemas.contracts import ConfirmationEvent, MessageEvent, ModelResponse
from aftersalesagent.tools.registry import ToolRegistry


def save(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", "utf-8")


class E2User(ScriptedUser):
    def __init__(self, plan, extra):
        super().__init__(plan)
        self.extra = extra
        self.item_answered = False

    def next_event(self, result):
        if self.extra.get("item_answer") and not self.item_answered:
            if result.session.pending_action:
                self.failures.append("proposal_before_item_clarification")
                self.finished = True
                return ConfirmationEvent(
                    event_id="reject-premature-item",
                    action_id=result.session.pending_action.action_id,
                    approved=False,
                )
            if (
                result.status not in {"failed", "truncated", "rejected", "handoff"}
                and re.search(
                    r"哪[件些个种]|具体.*商品|商品.*[编号名称]|which|what.*item|item.*(id|number)",
                    result.reply,
                    re.I,
                )
                and re.search(
                    r"[?？]|请|告诉|提供|告知|需要|could|please|provide|what|which",
                    result.reply,
                    re.I,
                )
            ):
                self.item_answered = True
                return self._message(self.extra["item_answer"], "answer_item_choice")
            self.failures.append("missing_item_question")
            self.finished = True
            return None
        return super().next_event(result)


class E2Replay(ScriptedNLReplay):
    """只用于离线驱动验证，可读用户计划，不能计为真实模型表现。"""

    def __init__(self, plan, scenario, extra):
        super().__init__(plan)
        self.scenario, self.extra = scenario, extra

    async def generate(self, request):
        users = [m for m in request.messages if m.role == "user"]
        if self.extra.get("item_answer") and len(users) == 1:
            return ModelResponse(content="请告诉我具体要退哪件商品？")
        if self.extra.get("item_answer") and len(users) > 1:
            observed = [
                json.loads(m.content).get("data") or {}
                for m in request.messages
                if m.role == "tool"
            ]
            if not any(d.get("user_id") == self.plan.source_user_id for d in observed):
                from aftersalesagent.schemas.contracts import ToolCall

                email = re.search(r"[A-Za-z0-9_.+%-]+@[A-Za-z0-9.-]+\.[A-Za-z]+", users[0].content)
                assert email
                self.calls += 1
                return ModelResponse(
                    tool_calls=[
                        ToolCall(
                            call_id=f"locate-{self.calls}",
                            name="find_user_id_by_email",
                            arguments={"email": email.group()},
                        )
                    ]
                )
        if self.scenario == "order_product":
            data = [
                json.loads(m.content).get("data") or {}
                for m in request.messages
                if m.role == "tool"
            ]
            if any(d.get("order_id") == self.plan.order_id for d in data):
                if not any(d.get("product_id") == self.extra["product_id"] for d in data):
                    self.calls += 1
                    from aftersalesagent.schemas.contracts import ToolCall

                    return ModelResponse(
                        tool_calls=[
                            ToolCall(
                                call_id=f"product-{self.calls}",
                                name="get_product_details",
                                arguments={"product_id": self.extra["product_id"]},
                            )
                        ]
                    )
                return ModelResponse(
                    content=f"订单{self.plan.order_id}状态pending，商品{self.extra['product_id']}规格已查询。"
                )
        return await super().generate(request)


async def business(case, *, folder, template, factory, policy, context, transport):
    folder.mkdir(parents=True, exist_ok=False)
    public = case["public"]
    plan = UserPlan.model_validate(case["user_script"]["plan"])
    extra = case["user_script"]["extra"]
    criteria = CaseCriteria.model_validate(case["private"]["criteria"])
    driver = E2User(plan, extra)
    store = clone_database(template, folder / "business.sqlite")
    service = RetailService(store)
    journal = Journal(folder / "application.sqlite")
    backend = factory()
    fault = public["scenario"] == "proposal_response_loss"
    marker = folder / "fault.once"
    provider = (
        MCPRetailProvider(
            service, trace_path=folder / "mcp.jsonl", fault_after_commit=marker if fault else None
        )
        if transport == "mcp"
        else None
    )
    registry = ToolRegistry(service, provider=provider)
    if fault and transport == "native":
        original = registry.invoke

        async def lose_response(call, context):
            result = await original(call, context)
            if (
                call.name
                in {
                    "cancel_pending_order",
                    "modify_pending_order_address",
                    "return_delivered_order_items",
                }
                and result.status == "ok"
                and not marker.exists()
            ):
                marker.write_text("committed; native response lost", "utf-8")
                raise ConnectionError("Injected native proposal response loss")
            return result

        registry.invoke = lose_response
    started = time.perf_counter()
    conversation, probe = [], None
    checks = {"plain_text_does_not_commit": True, "stale_confirmation_rejected": True}
    try:
        async with AsyncSqliteSaver.from_conn_string(str(folder / "checkpoints.sqlite")) as saver:
            runtime = LangGraphRuntime(
                backend,
                service,
                registry=registry,
                policy_text=policy,
                checkpointer=saver,
                journal=journal,
                trace_directory=folder,
            )
            session = await runtime.start(
                context.model_copy(
                    update={
                        "environment_id": store.path.stem,
                        "tool_transport": transport,
                    }
                )
            )
            sid = session.session_id
            initial = observe(store, sid)
            event = MessageEvent(event_id="user-1", content=public["first_user_message"])
            for _ in range(8):
                result = await runtime.advance(sid, event)
                conversation.append(
                    {
                        "event": event.model_dump(mode="json"),
                        "status": result.status,
                        "reply": result.reply,
                        "pending_action": result.session.pending_action.model_dump(mode="json")
                        if result.session.pending_action
                        else None,
                    }
                )
                if (
                    driver.said_yes
                    and isinstance(event, MessageEvent)
                    and (event.content == plan.plain_yes)
                ):
                    checks["plain_text_does_not_commit"] = (
                        observe(store, sid)["snapshot"] == initial["snapshot"]
                    )
                if isinstance(event, ConfirmationEvent):
                    break
                event = driver.next_event(result)
                if event is None:
                    break
            else:
                driver.failures.append("driver_event_limit")
            if driver.old_action_id:
                before = observe(store, sid)
                stale = await runtime.advance(
                    sid,
                    ConfirmationEvent(
                        event_id="stale-probe", action_id=driver.old_action_id, approved=True
                    ),
                )
                probe = {"status": stale.status, "reason": stale.termination_reason}
                checks["stale_confirmation_rejected"] = (
                    stale.status == "rejected"
                    and stale.termination_reason == "confirmation_stale"
                    and observe(store, sid) == before
                )
            events, final, usage = journal.events(sid), observe(store, sid), journal.usage(sid)
        verdict = verify(
            criteria, initial, final, events, result.reply, result.status, driver.failures
        ).model_dump(mode="json")
        verdict["checks"]["coarse_reply_evidence"] = reply_evidence(criteria, result.reply)
        verdict["checks"].update(checks)
        verdict["checks"].update(
            {
                "correction_exercised": not plan.correction or driver.corrected,
                "plain_yes_exercised": not plan.plain_yes or driver.said_yes,
                "item_clarification_exercised": not extra.get("item_answer")
                or driver.item_answered,
                "fault_exercised": not fault or marker.exists(),
            }
        )
        if extra.get("product_id"):

            def product_observed(event):
                data = event.payload.get("data")
                return (
                    event.kind == "tool_result"
                    and event.payload.get("status") == "ok"
                    and isinstance(data, dict)
                    and data.get("product_id") == extra["product_id"]
                )

            verdict["checks"]["product_read"] = any(product_observed(e) for e in events)
            verdict["checks"]["product_id_in_reply"] = extra["product_id"] in result.reply
        verdict["failure_labels"] = [k for k, v in verdict["checks"].items() if not v] + (
            driver.failures
        )
        verdict["success"] = all(verdict["checks"].values()) and not driver.failures
        request_count = sum(e.kind == "model_request" for e in events)
        response_count = sum(e.kind == "model_response" for e in events)
        row = {
            "task_id": public["task_id"],
            "scenario": public["scenario"],
            "kind": "business",
            "transport": transport,
            "verification": verdict,
            "terminal_status": result.status,
            "termination_reason": result.termination_reason,
            "usage": usage,
            "wall_seconds": time.perf_counter() - started,
            "driver_actions": driver.actions,
            "driver_failures": driver.failures,
            "stale_probe": probe,
            "fault_exercised": marker.exists(),
            "model_request_events": request_count,
            "model_response_events": response_count,
            "mcp_retry_events": sum(e.kind == "retry" for e in events),
            "trace": str(folder / "trace.jsonl"),
            "cost_estimate": None,
        }
        (folder / "trace.jsonl").write_text(
            "".join(e.model_dump_json() + "\n" for e in events), "utf-8"
        )
        save(folder / "conversation.json", conversation)
        save(folder / "initial_state.json", initial)
        save(folder / "final_state.json", final)
        save(folder / "result.json", row)
        return row
    finally:
        await registry.aclose()
        closer = getattr(backend, "aclose", None)
        if closer:
            await closer()
        journal.close()
        store.close()


class KnowledgeReplay:
    async def generate(self, request):
        payload = json.loads(request.messages[-1].content)
        c = next(c for c in payload["evidence"] if c["chunk_id"] == "R02")
        return ModelResponse(
            content=json.dumps(
                {
                    "answer": (
                        "退货需完好货品、原始购物凭证、已开具的发票；"
                        "银行卡退款需原始签购单和银行卡。"
                    ),
                    "status": "answered",
                    "citations": ["R02"],
                    "policy_versions": [c["version"]],
                },
                ensure_ascii=False,
            )
        )


async def knowledge(case, *, folder, root, template, settings, secret, transport, live):
    folder.mkdir(parents=True, exist_ok=False)
    settings = settings.model_copy(update={"knowledge_transport": transport})
    service = make_service(
        root,
        settings,
        secret,
        template=template,
        database=folder / "knowledge.sqlite",
        enable_paid_api=live,
        backend_override=None if live else KnowledgeReplay(),
    )
    try:
        question = KnowledgeQuestion(
            question=case["public"]["question"], mode="rag", **case["public"]["filters"]
        )
        started = time.perf_counter()
        raw = await service.ask(question, "policy-1")
        verdict = verify_knowledge(raw, case["private"]["criteria"])
        row = {
            "task_id": case["public"]["task_id"],
            "scenario": "policy",
            "kind": "knowledge",
            "transport": transport,
            "verification": verdict,
            "usage": raw["usage"],
            "wall_seconds": time.perf_counter() - started,
            "terminal_status": "completed" if raw["error"] is None else "failed",
            "raw_result": raw,
            "cost_estimate": None,
        }
        save(folder / "result.json", row)
        return row
    finally:
        await service.close()


def summary(rows):
    output = {}
    for mode in ("native", "mcp"):
        selected = [r for r in rows if r["transport"] == mode]
        output[mode] = {
            "episodes": len(selected),
            "passed": sum(r["verification"]["success"] for r in selected),
            "model_requests": sum(r["usage"]["model_calls"] for r in selected),
            "reported_tokens": sum(r["usage"]["reported_tokens"] or 0 for r in selected),
            "unknown_usage_calls": sum(r["usage"]["unknown_usage_calls"] for r in selected),
            "wall_seconds": sum(r["wall_seconds"] for r in selected),
            "business_passed": sum(
                r["kind"] == "business" and r["verification"]["success"] for r in selected
            ),
            "knowledge_passed": sum(
                r["kind"] == "knowledge" and r["verification"]["success"] for r in selected
            ),
            "cost_estimate": None,
        }
    return output
