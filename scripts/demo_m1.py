"""Deterministic offline demonstration, not a real LLM benchmark."""

import asyncio
import json
import uuid
from pathlib import Path

from aftersalesagent.business.database import import_pinned_database
from aftersalesagent.business.service import RetailService
from aftersalesagent.models.mock import MockBackend
from aftersalesagent.runtime.langgraph_runtime import LangGraphRuntime
from aftersalesagent.schemas.contracts import (
    ConfirmationEvent,
    MessageEvent,
    ModelResponse,
    RunContext,
    ToolCall,
)

ROOT = Path(__file__).resolve().parents[1]


async def main() -> None:
    run_id = "m1_demo_" + uuid.uuid4().hex[:10]
    store = import_pinned_database(ROOT, ROOT / ".local/databases" / (run_id + ".sqlite"))
    service = RetailService(store)
    backend = MockBackend(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        call_id="locate",
                        name="find_user_id_by_name_zip",
                        arguments={"first_name": "Yara", "last_name": "Muller", "zip": "85041"},
                    )
                ]
            ),
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        call_id="query",
                        name="get_order_details",
                        arguments={"order_id": "#W5056519"},
                    )
                ]
            ),
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        call_id="propose",
                        name="cancel_pending_order",
                        arguments={"order_id": "#W5056519", "reason": "ordered by mistake"},
                    )
                ]
            ),
        ]
    )
    policy = (ROOT / "data/upstream/tau2-bench/data/tau2/domains/retail/policy.md").read_text(
        "utf-8"
    )
    runtime = LangGraphRuntime(backend, service, policy_text=policy)
    try:
        view = await runtime.start(RunContext(environment_id=store.path.stem))
        turn = await runtime.advance(
            view.session_id,
            MessageEvent(
                event_id="user-1",
                content="I am Yara Muller in zip 85041. Cancel #W5056519; I ordered it by mistake.",
            ),
        )
        assert turn.status == "waiting_confirmation"
        action = turn.session.pending_action
        assert action is not None
        with store.transaction() as connection:
            before = store.order(connection, "#W5056519")
        assert before is not None and before["status"] == "pending"
        confirmed = await runtime.advance(
            view.session_id,
            ConfirmationEvent(event_id="user-2", action_id=action.action_id, approved=True),
        )
        with store.transaction() as connection:
            after = store.order(connection, "#W5056519")
        assert after is not None and after["status"] == "cancelled"
        assert confirmed.status == "completed"
        assert len(after["payment_history"]) == len(before["payment_history"]) + 1
        output = {
            "run_id": run_id,
            "mode": "scripted_mock_not_llm_evaluation",
            "database": str(store.path),
            "environment_id": store.path.stem,
            "before_confirmation": before["status"],
            "after_confirmation": after["status"],
            "model_calls": len(backend.requests),
            "confirmation_action_id": action.action_id,
            "source_commit": "fc0055dc4e0a316c3f83133267fbd6faaa770992",
            "agent_task_success_rate": None,
        }
        target = ROOT / ".local/results" / (run_id + ".json")
        target.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        trace = ROOT / ".local/traces" / (run_id + ".jsonl")
        trace.write_text(
            "".join(event.model_dump_json() + "\n" for event in turn.events + confirmed.events),
            encoding="utf-8",
        )
        print(json.dumps(output, indent=2))
        print("Trace:", trace)
    finally:
        store.close()


if __name__ == "__main__":
    asyncio.run(main())
