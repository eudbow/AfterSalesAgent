"""Real StateGraph + real SQLite + scripted model; no API calls."""

import asyncio
from pathlib import Path

from aftersalesagent.business.database import RetailStore
from aftersalesagent.business.service import RetailService
from aftersalesagent.models.mock import MockBackend
from aftersalesagent.runtime.langgraph_runtime import LangGraphRuntime
from aftersalesagent.schemas.contracts import (
    ConfirmationEvent,
    MessageEvent,
    ModelResponse,
    RunContext,
    RunLimits,
    ToolCall,
)


def response(name: str, **args) -> ModelResponse:
    return ModelResponse(tool_calls=[ToolCall(call_id=name, name=name, arguments=args)])


def policy() -> str:
    path = Path(__file__).resolve().parents[2] / (
        "data/upstream/tau2-bench/data/tau2/domains/retail/policy.md"
    )
    return path.read_text("utf-8")


def order(store: RetailStore, oid: str) -> dict:
    with store.transaction() as connection:
        result = store.order(connection, oid)
    assert result is not None
    return result


def test_query_error_feedback_is_visible_to_mock_and_can_recover(store: RetailStore) -> None:
    async def scenario():
        backend = MockBackend(
            [
                response("find_user_id_by_email", email="mia.garcia2723@example.com"),
                response("get_order_details", order_id="#Wnot-real"),
                response("get_order_details", order_id="#W5490111"),
                ModelResponse(content="The order is delivered."),
            ]
        )
        runtime = LangGraphRuntime(backend, RetailService(store), policy_text=policy())
        view = await runtime.start(RunContext(environment_id=store.path.stem))
        result = await runtime.advance(
            view.session_id, MessageEvent(event_id="1", content="Check my order")
        )
        assert result.status == "completed"
        tool_messages = [m for m in backend.requests[2].messages if m.role == "tool"]
        assert "not_found_or_forbidden" in tool_messages[-1].content
        assert "evaluation_criteria" not in backend.requests[0].model_dump_json()
        assert len([e for e in result.events if e.kind == "tool_call"]) == 3
        assert all(
            a.sequence < b.sequence for a, b in zip(result.events, result.events[1:], strict=False)
        )

    asyncio.run(scenario())


def test_cancel_confirmation_cycle_uses_service_not_model_claim(store: RetailStore) -> None:
    async def scenario():
        backend = MockBackend(
            [
                response(
                    "find_user_id_by_name_zip", first_name="Yara", last_name="Muller", zip="85041"
                ),
                response("get_order_details", order_id="#W5056519"),
                response("cancel_pending_order", order_id="#W5056519", reason="ordered by mistake"),
            ]
        )
        runtime = LangGraphRuntime(backend, RetailService(store), policy_text=policy())
        view = await runtime.start(RunContext(environment_id=store.path.stem))
        first = await runtime.advance(
            view.session_id, MessageEvent(event_id="1", content="Cancel my order.")
        )
        assert first.status == "waiting_confirmation"
        assert order(store, "#W5056519")["status"] == "pending"
        action = first.session.pending_action
        assert action is not None
        second = await runtime.advance(
            view.session_id,
            ConfirmationEvent(event_id="2", action_id=action.action_id, approved=True),
        )
        assert second.status == "completed"
        assert order(store, "#W5056519")["status"] == "cancelled"
        assert len(backend.requests) == 3
        replay = await runtime.advance(
            view.session_id,
            ConfirmationEvent(event_id="3", action_id=action.action_id, approved=True),
        )
        assert replay.status == "completed"
        assert len(order(store, "#W5056519")["payment_history"]) == 2

    asyncio.run(scenario())


def test_natural_language_yes_does_not_execute_old_confirmation(store: RetailStore) -> None:
    async def scenario():
        backend = MockBackend(
            [
                response(
                    "find_user_id_by_name_zip", first_name="Yara", last_name="Muller", zip="85041"
                ),
                response("cancel_pending_order", order_id="#W5056519", reason="ordered by mistake"),
                ModelResponse(content="Please confirm the current operation explicitly."),
            ]
        )
        service = RetailService(store)
        runtime = LangGraphRuntime(backend, service, policy_text=policy())
        view = await runtime.start(RunContext(environment_id=store.path.stem))
        first = await runtime.advance(
            view.session_id, MessageEvent(event_id="1", content="Cancel it")
        )
        assert first.session.pending_action is not None
        old = first.session.pending_action.action_id
        await runtime.advance(view.session_id, MessageEvent(event_id="2", content="yes"))
        assert order(store, "#W5056519")["status"] == "pending"
        assert service.confirm(view.session_id, old, True).status == "error"

    asyncio.run(scenario())


def test_runtime_can_submit_return_without_refunding_now(store: RetailStore) -> None:
    async def scenario():
        backend = MockBackend(
            [
                response("find_user_id_by_email", email="mia.garcia2723@example.com"),
                response(
                    "return_delivered_order_items",
                    order_id="#W5490111",
                    item_ids=["1421289881"],
                    payment_method_id="credit_card_3124723",
                ),
            ]
        )
        runtime = LangGraphRuntime(backend, RetailService(store), policy_text=policy())
        view = await runtime.start(RunContext(environment_id=store.path.stem))
        before = order(store, "#W5490111")
        first = await runtime.advance(
            view.session_id, MessageEvent(event_id="1", content="Return the keyboard")
        )
        assert first.session.pending_action is not None
        await runtime.advance(
            view.session_id,
            ConfirmationEvent(
                event_id="2", action_id=first.session.pending_action.action_id, approved=True
            ),
        )
        after = order(store, "#W5490111")
        assert after["status"] == "return requested"
        assert after["payment_history"] == before["payment_history"]

    asyncio.run(scenario())


def test_mock_exhaustion_is_an_explicit_backend_failure(store: RetailStore) -> None:
    async def scenario():
        runtime = LangGraphRuntime(MockBackend([]), RetailService(store), policy_text=policy())
        view = await runtime.start(RunContext(environment_id=store.path.stem))
        result = await runtime.advance(view.session_id, MessageEvent(event_id="1", content="Check"))
        assert result.status == "failed" and result.termination_reason == "model_backend_error"

    asyncio.run(scenario())


def test_model_call_limit_stops_loop(store: RetailStore) -> None:
    async def scenario():
        backend = MockBackend([response("calculate", expression="1+1") for _ in range(4)])
        runtime = LangGraphRuntime(backend, RetailService(store), policy_text=policy())
        view = await runtime.start(
            RunContext(environment_id=store.path.stem, limits=RunLimits(max_model_calls=2))
        )
        result = await runtime.advance(view.session_id, MessageEvent(event_id="1", content="Loop"))
        assert result.status == "truncated" and result.termination_reason == "model_call_limit"
        assert len(backend.requests) == 2

    asyncio.run(scenario())


def test_tool_call_limit_stops_before_second_tool(store: RetailStore) -> None:
    async def scenario():
        backend = MockBackend([response("calculate", expression="1+1") for _ in range(3)])
        runtime = LangGraphRuntime(backend, RetailService(store), policy_text=policy())
        view = await runtime.start(
            RunContext(environment_id=store.path.stem, limits=RunLimits(max_tool_calls=1))
        )
        result = await runtime.advance(view.session_id, MessageEvent(event_id="1", content="Loop"))
        assert result.status == "truncated" and result.termination_reason == "tool_call_limit"
        assert len([e for e in result.events if e.kind == "tool_result"]) == 1

    asyncio.run(scenario())


def test_multiple_model_tool_calls_are_rejected_without_execution(store: RetailStore) -> None:
    async def scenario():
        calls = [
            ToolCall(call_id=str(i), name="calculate", arguments={"expression": "1+1"})
            for i in range(2)
        ]
        runtime = LangGraphRuntime(
            MockBackend([ModelResponse(tool_calls=calls)]),
            RetailService(store),
            policy_text=policy(),
        )
        view = await runtime.start(RunContext(environment_id=store.path.stem))
        result = await runtime.advance(view.session_id, MessageEvent(event_id="1", content="Run"))
        assert result.termination_reason == "multiple_tool_calls"
        assert not any(e.kind == "tool_result" for e in result.events)

    asyncio.run(scenario())


def test_active_timeout_and_reported_token_limit(store: RetailStore) -> None:
    class SlowBackend(MockBackend):
        async def generate(self, request):
            await asyncio.sleep(0.1)
            return ModelResponse(content="late")

    async def scenario():
        runtime = LangGraphRuntime(SlowBackend([]), RetailService(store), policy_text=policy())
        view = await runtime.start(
            RunContext(environment_id=store.path.stem, limits=RunLimits(max_active_seconds=0.01))
        )
        timed = await runtime.advance(view.session_id, MessageEvent(event_id="1", content="Run"))
        assert timed.status == "truncated" and timed.termination_reason == "active_time_limit"
        over = ModelResponse(
            tool_calls=[ToolCall(call_id="x", name="calculate", arguments={"expression": "1+1"})],
            input_tokens=11,
            output_tokens=1,
        )
        runtime2 = LangGraphRuntime(MockBackend([over]), RetailService(store), policy_text=policy())
        view2 = await runtime2.start(
            RunContext(environment_id=store.path.stem, limits=RunLimits(max_tokens=10))
        )
        result = await runtime2.advance(view2.session_id, MessageEvent(event_id="2", content="Run"))
        assert result.termination_reason == "reported_token_limit"
        assert not any(e.kind == "tool_call" for e in result.events)

    asyncio.run(scenario())


def test_timeout_budget_cannot_be_reset_by_another_message(store: RetailStore) -> None:
    class SlowBackend(MockBackend):
        attempts = 0

        async def generate(self, request):
            self.attempts += 1
            await asyncio.sleep(0.5)
            return ModelResponse(content="late")

    async def scenario():
        backend = SlowBackend([])
        runtime = LangGraphRuntime(backend, RetailService(store), policy_text=policy())
        view = await runtime.start(
            RunContext(environment_id=store.path.stem, limits=RunLimits(max_active_seconds=0.1))
        )
        first = await runtime.advance(view.session_id, MessageEvent(event_id="1", content="Run"))
        second = await runtime.advance(
            view.session_id, MessageEvent(event_id="2", content="Try again")
        )
        assert first.status == second.status == "truncated"
        assert backend.attempts == 1

    asyncio.run(scenario())
