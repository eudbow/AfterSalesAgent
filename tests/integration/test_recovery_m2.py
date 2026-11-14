"""持久化预算和工具准备重放的故障验证。"""

import asyncio
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from sqlalchemy import func, select

from aftersalesagent.business.database import RetailStore, pending
from aftersalesagent.business.service import RetailService
from aftersalesagent.harness.journal import EventConflict, Journal
from aftersalesagent.models.mock import MockBackend
from aftersalesagent.runtime.langgraph_runtime import LangGraphRuntime
from aftersalesagent.schemas.contracts import (
    MessageEvent,
    ModelResponse,
    RunContext,
    RunLimits,
    ToolCall,
)


def call(name, **args):
    return ModelResponse(tool_calls=[ToolCall(call_id=name, name=name, arguments=args)])


def test_tool_preparation_response_loss_does_not_create_new_action(
    store: RetailStore,
    monkeypatch,
):
    service = RetailService(store)
    original = service.dispatch
    lost = []

    def lose_once(sid, tool, **kwargs):
        result = original(sid, tool, **kwargs)
        if tool.name == "cancel_pending_order" and not lost:
            lost.append(result)
            raise TimeoutError("Injected after prepared action committed")
        return result

    monkeypatch.setattr(service, "dispatch", lose_once)

    async def scenario():
        runtime = LangGraphRuntime(
            MockBackend(
                [
                    call(
                        "find_user_id_by_name_zip",
                        first_name="Yara",
                        last_name="Muller",
                        zip="85041",
                    ),
                    call("cancel_pending_order", order_id="#W5056519", reason="ordered by mistake"),
                ]
            ),
            service,
            policy_text="policy",
        )
        session = await runtime.start(RunContext(environment_id=store.path.stem))
        result = await runtime.advance(
            session.session_id, MessageEvent(event_id="m", content="cancel")
        )
        assert result.status == "waiting_confirmation"
        assert result.session.pending_action is not None
        assert (
            result.session.pending_action.action_id == lost[0].data["pending_action"]["action_id"]
        )
        with store.transaction() as connection:
            assert connection.scalar(select(func.count()).select_from(pending)) == 1

    asyncio.run(scenario())


def test_total_and_per_turn_budgets_survive_reopening(
    store: RetailStore,
    tmp_path: Path,
):
    async def scenario():
        journal_path, checkpoint = tmp_path / "journal.sqlite", tmp_path / "checkpoints.sqlite"
        journal = Journal(journal_path)
        service = RetailService(store)
        first_backend = MockBackend([call("calculate", expression="1+1")])
        async with AsyncSqliteSaver.from_conn_string(str(checkpoint)) as saver:
            runtime = LangGraphRuntime(
                first_backend, service, policy_text="policy", checkpointer=saver, journal=journal
            )
            view = await runtime.start(
                RunContext(
                    environment_id=store.path.stem,
                    limits=RunLimits(max_model_calls=2, max_turn_model_calls=1),
                )
            )
            first = await runtime.advance(
                view.session_id, MessageEvent(event_id="1", content="calculate")
            )
            assert first.termination_reason == "turn_model_call_limit"
            assert journal.usage(view.session_id)["model_calls"] == 1
        journal.close()
        journal = Journal(journal_path)
        async with AsyncSqliteSaver.from_conn_string(str(checkpoint)) as saver:
            backend = MockBackend(
                [ModelResponse(content="second"), ModelResponse(content="should not run")]
            )
            reopened = LangGraphRuntime(
                backend, service, policy_text="policy", checkpointer=saver, journal=journal
            )
            second = await reopened.advance(
                view.session_id, MessageEvent(event_id="2", content="continue")
            )
            assert second.status == "completed"
            third = await reopened.advance(
                view.session_id, MessageEvent(event_id="3", content="again")
            )
            assert third.termination_reason == "model_call_limit"
            assert len(backend.requests) == 1
            assert any(
                m.role == "user" and m.content == "calculate" for m in backend.requests[0].messages
            )
        journal.close()

    asyncio.run(scenario())


def test_crash_reservation_is_conservative_and_unknown_usage_stays_unknown(tmp_path: Path):
    path = tmp_path / "budget.sqlite"
    j = Journal(path)
    j.register("session")
    event = MessageEvent(event_id="request", content="message")
    j.claim("session", event)
    limits = RunLimits(max_active_seconds=60, max_turn_active_seconds=10)
    _, seconds = j.reserve("session", "request", limits)
    assert seconds == 10
    j.charge("session", "request", "model_calls")
    j.charge("session", "request", "unknown_usage_calls")
    j.charge("session", "request", "tokens", 12)
    j.close()
    reopened = Journal(path)
    assert reopened.usage("session")["active_seconds"] == 10
    assert reopened.usage("session")["reported_tokens"] == 12
    assert reopened.usage("session")["total_tokens"] is None
    _, available = reopened.reserve("session", "request", limits)
    assert available == 0
    try:
        reopened.claim("session", MessageEvent(event_id="other", content="new"))
    except EventConflict:
        pass
    else:
        raise AssertionError("Unresolved request was bypassed")
    reopened.close()


def test_trace_redacts_structured_credentials_and_exports_read_only(tmp_path: Path):
    from datetime import UTC, datetime

    journal = Journal()
    sid = "a" * 32
    journal.append(
        sid,
        "model_response",
        {
            "provider_metadata": {"Authorization": "Bearer secret", "api_key": "secret"},
            "nested": [{"access_token": "secret"}],
            "content": "public",
        },
        datetime.now(UTC),
    )
    text = journal.export(sid, tmp_path).read_text("utf-8")
    assert "secret" not in text and "[REDACTED]" in text and "public" in text
    journal.close()


def test_simultaneous_duplicate_event_runs_once(store: RetailStore):
    async def scenario():
        backend = MockBackend([ModelResponse(content="one reply")])
        runtime = LangGraphRuntime(backend, RetailService(store), policy_text="policy")
        session = await runtime.start(RunContext(environment_id=store.path.stem))
        event = MessageEvent(event_id="same", content="same")
        first, second = await asyncio.gather(
            runtime.advance(session.session_id, event),
            runtime.advance(session.session_id, event),
        )
        assert first == second
        assert len(backend.requests) == 1
        assert runtime.journal.usage(session.session_id)["model_calls"] == 1

    asyncio.run(scenario())


def test_model_timeout_keeps_token_usage_unknown(store: RetailStore):
    class TimeoutBackend:
        async def generate(self, request):
            await asyncio.sleep(1)
            return ModelResponse(content="late")

    async def scenario():
        runtime = LangGraphRuntime(TimeoutBackend(), RetailService(store), policy_text="policy")
        session = await runtime.start(
            RunContext(
                environment_id=store.path.stem,
                limits=RunLimits(max_active_seconds=0.1),
            )
        )
        result = await runtime.advance(
            session.session_id, MessageEvent(event_id="m", content="run")
        )
        assert result.status == "truncated"
        usage = runtime.journal.usage(session.session_id)
        assert usage["model_calls"] == 1
        assert usage["total_tokens"] is None
        assert usage["unknown_usage_calls"] == 1

    asyncio.run(scenario())
