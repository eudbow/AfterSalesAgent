"""固定 S0/S1 的新开发诊断：私有计划仅驱动用户，不进入真实模型请求。"""

import json
import re
import time
from collections import Counter

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from aftersalesagent.business.database import clone_database
from aftersalesagent.business.service import RetailService
from aftersalesagent.evaluation.suite import CaseCriteria
from aftersalesagent.evaluation.verifier import observe, verify
from aftersalesagent.harness.journal import Journal
from aftersalesagent.mcp_transport.adapters import MCPRetailProvider
from aftersalesagent.runtime.langgraph_runtime import LangGraphRuntime
from aftersalesagent.schemas.contracts import (
    ConfirmationEvent,
    MessageEvent,
    ModelResponse,
    ToolCall,
)
from aftersalesagent.skills.registry import SkillRegistry
from aftersalesagent.tools.registry import ToolRegistry


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalise(arguments):
    output = dict(arguments)
    if "item_ids" in output:
        output["item_ids"] = sorted(output["item_ids"])
    return output


def matches(action, tool, arguments):
    return action.tool_name == tool and normalise(action.arguments) == normalise(arguments)


def asks(reply, slot):
    words = {
        "zip": r"邮编|邮政编码|zip|postal",
        "items": r"商品|哪.*件|item|product",
        "reason": r"原因|理由|reason|why",
        "payment": r"退款.*方式|退款.*哪|退.*到|支付方式|payment|refund.*method",
    }
    return bool(
        re.search(words[slot], reply, re.I)
        and re.search(
            r"[?？]|请|提供|告诉|告知|需要|选择|could|please|provide|which|what|need", reply, re.I
        )
    )


class WorkflowUser:
    def __init__(self, plan):
        self.plan = plan
        self.index = 0
        self.failures = []
        self.actions = []
        self.old_actions = []
        self.finished = False

    def reject(self, result, label):
        self.failures.append(label)
        self.finished = True
        if result.session.pending_action:
            return ConfirmationEvent(
                event_id="reject-unexpected",
                action_id=result.session.pending_action.action_id,
                approved=False,
            )
        return None

    def next_event(self, result):
        if self.finished:
            return None
        if result.status in {"failed", "truncated", "handoff", "rejected"}:
            self.finished = True
            return None
        action = result.session.pending_action
        if self.index < len(self.plan["steps"]):
            step = self.plan["steps"][self.index]
            if step["trigger"] == "proposal":
                if not action:
                    return self.reject(result, "expected_proposal_missing:" + step["label"])
                if not matches(action, step["expected_tool"], step["expected_arguments"]):
                    return self.reject(result, "unexpected_stage_arguments:" + step["label"])
                self.old_actions.append(action.action_id)
            elif step["trigger"] == "question":
                if action:
                    return self.reject(result, "proposal_before_clarification:" + step["label"])
                if not asks(result.reply, step["slot"]):
                    return self.reject(result, "required_question_missing:" + step["label"])
            elif step["trigger"] == "refusal":
                if action or not re.search(
                    r"cannot|can't|unable|not (?:allowed|possible)|无法|不能|不支持|不允许",
                    result.reply,
                    re.I,
                ):
                    return self.reject(result, "required_refusal_missing")
            else:
                raise ValueError("Unsupported user trigger")
            self.index += 1
            self.actions.append({"step": step["label"], "message": step["reply"]})
            return MessageEvent(event_id=f"user-{self.index + 1}", content=step["reply"])
        if action is None:
            return self.reject(result, "final_proposal_missing")
        final = self.plan["final"]
        if not matches(action, final["tool_name"], final["arguments"]):
            return self.reject(result, "unexpected_final_arguments")
        if action.action_id in self.old_actions:
            return self.reject(result, "stale_card_reused")
        self.finished = True
        self.actions.append({"step": "formal_confirmation", "approved": True})
        return ConfirmationEvent(
            event_id="formal-confirmation", action_id=action.action_id, approved=True
        )


class WorkflowReplay:
    """明确可读未来用户计划的离线脚本模型，仅验证驱动器，不作能力评分。"""

    def __init__(self, plan):
        self.plan = plan
        self.calls = 0

    async def generate(self, request):
        users = [m for m in request.messages if m.role == "user"]
        turn = len(users) - 1
        step = self.plan["steps"][turn] if turn < len(self.plan["steps"]) else None
        if step and step["trigger"] == "question":
            question = {
                "zip": "请提供邮编？",
                "items": "请告诉我退哪些商品编号？",
                "reason": "请提供取消原因？",
                "payment": "请选择退款方式？",
            }
            return ModelResponse(content=question[step["slot"]])
        data = [
            json.loads(m.content).get("data") or {} for m in request.messages if m.role == "tool"
        ]
        self.calls += 1
        if not any(d.get("user_id") == self.plan["source_user_id"] for d in data):
            email = re.search(r"[A-Za-z0-9_.+%-]+@[A-Za-z0-9.-]+\.[A-Za-z]+", users[0].content)
            assert email
            name, args = "find_user_id_by_email", {"email": email.group()}
        elif not any(d.get("order_id") == self.plan["order_id"] for d in data):
            name, args = "get_order_details", {"order_id": self.plan["order_id"]}
        elif step and step["trigger"] == "refusal":
            return ModelResponse(content=f"订单{self.plan['order_id']}已经送达，不能取消。")
        elif step:
            name, args = step["expected_tool"], step["expected_arguments"]
        else:
            name, args = self.plan["final"]["tool_name"], self.plan["final"]["arguments"]
        return ModelResponse(
            tool_calls=[
                ToolCall.model_validate(
                    {"call_id": f"replay-{self.calls}", "name": name, "arguments": args}
                )
            ]
        )


def process_metrics(events):
    invalid = Counter()
    errors = Counter()
    free_replies = []
    input_tokens = output_tokens = 0
    for i, event in enumerate(events):
        payload = event.payload
        if event.kind == "tool_result" and payload.get("status") == "error":
            code = payload["error"]["code"]
            errors[code] += 1
            if code == "invalid_arguments":
                preceding = [
                    e
                    for e in events[:i]
                    if e.kind == "tool_call" and e.payload["call_id"] == payload["call_id"]
                ]
                if preceding:
                    call = preceding[-1].payload
                    invalid[json.dumps([call["name"], call["arguments"]], sort_keys=True)] += 1
        if event.kind == "model_response":
            input_tokens += payload.get("input_tokens") or 0
            output_tokens += payload.get("output_tokens") or 0
            if not payload.get("tool_calls") and str(payload.get("content", "")).strip():
                text = str(payload["content"])
                free_replies.append(
                    {
                        "sequence": event.sequence,
                        "content": text,
                        "contains_chinese": bool(re.search(r"[\u4e00-\u9fff]", text)),
                    }
                )
    return {
        "tool_errors": dict(errors),
        "invalid_arguments": errors["invalid_arguments"],
        "repeated_identical_invalid_calls": sum(max(0, n - 1) for n in invalid.values()),
        "free_model_replies": free_replies,
        "non_chinese_reply_flags": sum(not x["contains_chinese"] for x in free_replies),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


async def run_case(case, *, folder, template, factory, policy, context, group, repetition, root):
    folder.mkdir(parents=True, exist_ok=False)
    store = clone_database(template, folder / "business.sqlite")
    journal = Journal(folder / "application.sqlite")
    service = RetailService(store)
    backend = factory()
    provider = MCPRetailProvider(service, trace_path=folder / "mcp.jsonl")
    registry = ToolRegistry(service, provider=provider)
    skill_registry = None
    if group == "S1":
        skill_registry = SkillRegistry(folder / "skills.sqlite")
        skill_registry.install(root / "skills", "after-sales-workflow", "1.0.0")
    driver = WorkflowUser(case["plan"])
    conversation = []
    unconfirmed_unchanged = True
    started = time.perf_counter()
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
                skill_provider=skill_registry,
            )
            context = type(context).model_validate(
                context.model_dump()
                | {
                    "environment_id": store.path.stem,
                    "tool_transport": "mcp",
                    "skill_id": "after-sales-workflow" if group == "S1" else None,
                    "skill_version": "1.0.0" if group == "S1" else "none",
                }
            )
            session = await runtime.start(context)
            sid = session.session_id
            initial = observe(store, sid)
            event = MessageEvent(event_id="user-1", content=case["public"]["first_user_message"])
            for _ in range(case["plan"]["max_events"]):
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
                if isinstance(event, MessageEvent):
                    unconfirmed_unchanged &= observe(store, sid)["snapshot"] == initial["snapshot"]
                if isinstance(event, ConfirmationEvent):
                    break
                event = driver.next_event(result)
                if event is None:
                    break
            else:
                driver.failures.append("driver_event_limit")
            events, final, usage = journal.events(sid), observe(store, sid), journal.usage(sid)
        goal = CaseCriteria.model_validate(case["criteria"])
        # 商品集合的顺序无业务含义；只在评分副本归一化，不改原轨迹或服务参数。
        goal = goal.model_copy(update={"arguments": normalise(goal.arguments)})
        scoring_events = [
            e.model_copy(
                update={"payload": e.payload | {"arguments": normalise(e.payload["arguments"])}}
            )
            if e.kind == "pending_action"
            else e
            for e in events
        ]
        verdict = verify(
            goal, initial, final, scoring_events, result.reply, result.status, driver.failures
        ).model_dump(mode="json")
        verdict["checks"]["all_user_stages_exercised"] = driver.index == len(case["plan"]["steps"])
        verdict["checks"]["no_unconfirmed_business_changes"] = unconfirmed_unchanged
        verdict["failure_labels"] = [
            k for k, value in verdict["checks"].items() if not value
        ] + driver.failures
        verdict["success"] = all(verdict["checks"].values()) and not driver.failures
        metrics = process_metrics(events)
        row = {
            "task_id": case["public"]["task_id"],
            "family": case["plan"]["family"],
            "title": case["plan"]["title"],
            "group": group,
            "repetition": repetition,
            "verification": verdict,
            "terminal_status": result.status,
            "termination_reason": result.termination_reason,
            "usage": usage,
            "process": metrics,
            "driver_actions": driver.actions,
            "driver_failures": driver.failures,
            "wall_seconds": time.perf_counter() - started,
            "trace": str(folder / "trace.jsonl"),
            "cost_estimate": None,
        }
        (folder / "trace.jsonl").write_text(
            "".join(e.model_dump_json() + "\n" for e in events), encoding="utf-8"
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


def aggregate(rows):
    groups = {}
    for group in ("S0", "S1"):
        chosen = [r for r in rows if r["group"] == group]
        groups[group] = {
            "episodes": len(chosen),
            "passed": sum(r["verification"]["success"] for r in chosen),
            "model_calls": sum(r["usage"]["model_calls"] for r in chosen),
            "input_tokens": sum(r["process"]["input_tokens"] for r in chosen),
            "output_tokens": sum(r["process"]["output_tokens"] for r in chosen),
            "reported_tokens": sum(r["usage"]["reported_tokens"] for r in chosen),
            "unknown_usage_calls": sum(r["usage"]["unknown_usage_calls"] for r in chosen),
            "invalid_arguments": sum(r["process"]["invalid_arguments"] for r in chosen),
            "repeated_invalid_calls": sum(
                r["process"]["repeated_identical_invalid_calls"] for r in chosen
            ),
            "non_chinese_reply_flags": sum(r["process"]["non_chinese_reply_flags"] for r in chosen),
            "wall_seconds": sum(r["wall_seconds"] for r in chosen),
            "cost_estimate": None,
        }
    paired = []
    for task_id, repetition in sorted({(r["task_id"], r["repetition"]) for r in rows}):
        pair = {
            r["group"]: r for r in rows if r["task_id"] == task_id and r["repetition"] == repetition
        }
        if len(pair) == 2:
            s0, s1 = (pair[g]["verification"]["success"] for g in ("S0", "S1"))
            paired.append(
                {
                    "task_id": task_id,
                    "repetition": repetition,
                    "S0": s0,
                    "S1": s1,
                    "comparison": "both_pass"
                    if s0 and s1
                    else "both_fail"
                    if not s0 and not s1
                    else "S1_improved"
                    if s1
                    else "S1_regressed",
                }
            )
    return {
        "groups": groups,
        "paired": paired,
        "paired_counts": dict(Counter(p["comparison"] for p in paired)),
    }
