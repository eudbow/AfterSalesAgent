"""用户工厂显式注入，业务运行与事后评分分离；不修改任何模块全局变量。"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from aftersalesagent.business.database import clone_database
from aftersalesagent.business.service import RetailService
from aftersalesagent.evaluation.e3_dev_runner import save
from aftersalesagent.evaluation.e3_v2.contracts import UserFactory
from aftersalesagent.evaluation.e3_v2.review import fingerprint
from aftersalesagent.evaluation.e3_v2.scoring import score
from aftersalesagent.evaluation.e3_v2.user import FormalUser
from aftersalesagent.evaluation.verifier import observe
from aftersalesagent.harness.journal import Journal
from aftersalesagent.mcp_transport.adapters import MCPRetailProvider
from aftersalesagent.models.base import ModelBackend
from aftersalesagent.runtime.langgraph_runtime import LangGraphRuntime
from aftersalesagent.schemas.contracts import ConfirmationEvent, MessageEvent, RunContext
from aftersalesagent.skills.base import SkillProvider
from aftersalesagent.tools.registry import ToolRegistry


async def run_case(
    case,
    *,
    folder: Path,
    template: Path,
    backend_factory: Callable[[], ModelBackend],
    policy_text: str,
    context: RunContext,
    protocol: dict,
    user_factory: UserFactory = FormalUser,
    skill_provider: SkillProvider | None = None,
    group: str = "S0",
    transport: str = "mcp",
):
    if transport not in {"native", "mcp"}:
        raise ValueError("Unknown tool transport")
    if group == "S0" and context.skill_id is not None:
        raise ValueError("S0 must keep the complete core policy without a Skill")
    if group == "S1" and (context.skill_id, context.skill_version) != (
        "after-sales-workflow",
        "1.0.0",
    ):
        raise ValueError("S1 must select the original fixed Skill v1")
    driver = user_factory(case["script"])
    folder.mkdir(parents=True, exist_ok=False)
    store = clone_database(template, folder / "business.sqlite")
    journal = Journal(folder / "application.sqlite")
    backend = backend_factory()
    service = RetailService(store)
    registry = ToolRegistry(
        service,
        provider=MCPRetailProvider(service, trace_path=folder / "mcp.jsonl")
        if transport == "mcp"
        else None,
    )
    conversation = []
    unchanged = True
    started = time.perf_counter()
    try:
        async with AsyncSqliteSaver.from_conn_string(str(folder / "checkpoints.sqlite")) as saver:
            runtime = LangGraphRuntime(
                backend,
                service,
                registry=registry,
                policy_text=policy_text,
                checkpointer=saver,
                journal=journal,
                trace_directory=folder,
                skill_provider=skill_provider,
            )
            selected = RunContext.model_validate(
                context.model_dump()
                | {"environment_id": store.path.stem, "tool_transport": transport}
            )
            sid = (await runtime.start(selected)).session_id
            initial = observe(store, sid)
            event = MessageEvent(event_id="v2-initial", content=case["public"].first_user_message)
            for _ in range(case["script"].max_events):
                language = driver.state().language if isinstance(event, MessageEvent) else None
                result = await runtime.advance(sid, event)
                conversation.append(
                    {
                        "event": event.model_dump(mode="json"),
                        "input_language": language,
                        "status": result.status,
                        "reply": result.reply,
                        "pending_action": result.session.pending_action.model_dump(mode="json")
                        if result.session.pending_action
                        else None,
                    }
                )
                if isinstance(event, MessageEvent):
                    unchanged &= observe(store, sid)["snapshot"] == initial["snapshot"]
                if isinstance(event, ConfirmationEvent):
                    break
                event = driver.next_event(result)
                if event is None:
                    break
            else:
                driver.on_limit()
            final = observe(store, sid)
            events, usage = journal.events(sid), journal.usage(sid)
        scoring = score(
            criteria=case["criteria"],
            initial=initial,
            final=final,
            events=events,
            reply=result.reply,
            terminal=result.status,
            simulator=driver.state(),
            no_unconfirmed_changes=unchanged,
            conversation=conversation,
            protocol=protocol,
        )
        row = {
            "task_id": case["public"].task_id,
            "group": group,
            "transport": transport,
            "protocol_id": protocol["protocol_id"],
            "protocol_version": protocol["version"],
            "protocol_fingerprint": fingerprint(protocol),
            "scoring": scoring,
            "user_simulator": driver.describe(),
            "usage": usage,
            "terminal_status": result.status,
            "termination_reason": result.termination_reason,
            "wall_seconds": time.perf_counter() - started,
            "cost_estimate": None,
        }
        (folder / "trace.jsonl").write_text(
            "".join(event.model_dump_json() + "\n" for event in events), encoding="utf-8"
        )
        for name, value in (
            ("conversation", conversation),
            ("initial_state", initial),
            ("final_state", final),
            ("result", row),
        ):
            save(folder / (name + ".json"), value)
        return row
    finally:
        await registry.aclose()
        closer = getattr(backend, "aclose", None)
        if closer:
            await closer()
        journal.close()
        store.close()
