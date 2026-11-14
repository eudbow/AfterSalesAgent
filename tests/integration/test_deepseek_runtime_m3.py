"""真实Runtime/业务环境 + HTTP Mock；不是付费模型评测。"""

import asyncio
import json

import httpx
import pytest
from pydantic import SecretStr

from aftersalesagent.api.app import create_app
from aftersalesagent.business.service import RetailService
from aftersalesagent.config import Settings
from aftersalesagent.models.deepseek import DeepSeekBackend, DeepSeekOptions
from aftersalesagent.runtime.langgraph_runtime import LangGraphRuntime
from aftersalesagent.schemas.contracts import ConfirmationEvent, MessageEvent, RunContext

SECRET = "offline-secret"


def payload(name=None, args=None, reason=None, content=None):
    return {
        "model": "test",
        "id": "test-response",
        "choices": [
            {
                "finish_reason": reason or ("tool_calls" if name else "stop"),
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": (
                        [
                            {
                                "id": name,
                                "type": "function",
                                "function": {"name": name, "arguments": json.dumps(args or {})},
                            }
                        ]
                        if name
                        else []
                    ),
                },
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 5},
    }


def test_deepseek_http_mock_drives_confirmation_without_key_leak(store):
    seen = []
    responses = [
        payload(
            "find_user_id_by_name_zip",
            {"first_name": "Yara", "last_name": "Muller", "zip": "85041"},
        ),
        payload("get_order_details", {"order_id": "#W5056519"}),
        payload("cancel_pending_order", {"order_id": "#W5056519", "reason": "ordered by mistake"}),
    ]

    def respond(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=responses.pop(0))

    async def scenario():
        backend = DeepSeekBackend(
            DeepSeekOptions(model="test"), SecretStr(SECRET), transport=httpx.MockTransport(respond)
        )
        runtime = LangGraphRuntime(
            backend, RetailService(store), policy_text="Use one tool at a time."
        )
        session = await runtime.start(
            RunContext(environment_id=store.path.stem, backend="deepseek", model_name="test")
        )
        first = await runtime.advance(
            session.session_id, MessageEvent(event_id="m", content="Cancel order")
        )
        assert first.status == "waiting_confirmation"
        assert first.session.pending_action is not None
        confirmed = await runtime.advance(
            session.session_id,
            ConfirmationEvent(
                event_id="c", action_id=first.session.pending_action.action_id, approved=True
            ),
        )
        assert confirmed.status == "completed"
        usage = runtime.journal.usage(session.session_id)
        assert usage["total_tokens"] == 75 and usage["unknown_usage_calls"] == 0
        assert any(m["role"] == "tool" for m in seen[-1]["messages"])
        assert SECRET not in "\n".join(
            e.model_dump_json() for e in runtime.journal.events(session.session_id)
        )
        assert "criteria" not in json.dumps(seen)
        await backend.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "mode,expected,status,known",
    [
        ("auth", "model_authentication_failed", "failed", False),
        ("bad_json", "model_protocol_error", "failed", True),
        ("length", "model_output_limit", "truncated", True),
    ],
)
def test_runtime_records_api_failures_and_usage_without_business_execution(
    store, mode, expected, status, known
):
    def respond(_):
        if mode == "auth":
            return httpx.Response(401, text=SECRET)
        data = payload(
            "cancel_pending_order",
            {"order_id": "#W5056519"},
            reason=("length" if mode == "length" else "tool_calls"),
        )
        data["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = "{incomplete"
        return httpx.Response(200, json=data)

    async def scenario():
        backend = DeepSeekBackend(
            DeepSeekOptions(model="test"), SecretStr(SECRET), transport=httpx.MockTransport(respond)
        )
        runtime = LangGraphRuntime(backend, RetailService(store), policy_text="policy")
        session = await runtime.start(
            RunContext(environment_id=store.path.stem, backend="deepseek", model_name="test")
        )
        result = await runtime.advance(
            session.session_id, MessageEvent(event_id="m", content="message")
        )
        assert result.status == status and result.termination_reason == expected
        assert not any(event.kind == "tool_call" for event in result.events)
        usage = runtime.journal.usage(session.session_id)
        assert (usage["total_tokens"] == 25) if known else usage["total_tokens"] is None
        assert SECRET not in result.model_dump_json()
        await backend.aclose()

    asyncio.run(scenario())


def test_web_live_mode_requires_explicit_enable_before_reading_credentials():
    with pytest.raises(ValueError, match="enable-paid-api"):
        create_app(settings=Settings(backend="deepseek", model_name="test"))


def test_budget_stop_closes_tool_protocol_before_next_user_turn(store):
    from aftersalesagent.schemas.contracts import RunLimits

    first = payload("calculate", {"expression": "1+1"})
    first["usage"] = {"prompt_tokens": 5, "completion_tokens": 1}
    second = payload(content="Recovered safely")
    second["usage"] = {"prompt_tokens": 1, "completion_tokens": 1}
    responses = [first, second]
    seen = []

    def respond(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=responses.pop(0))

    async def scenario():
        backend = DeepSeekBackend(
            DeepSeekOptions(model="test"), SecretStr(SECRET), transport=httpx.MockTransport(respond)
        )
        runtime = LangGraphRuntime(backend, RetailService(store), policy_text="policy")
        session = await runtime.start(
            RunContext(
                environment_id=store.path.stem,
                backend="deepseek",
                model_name="test",
                limits=RunLimits(max_tokens=12, max_turn_tokens=6),
            )
        )
        stopped = await runtime.advance(
            session.session_id, MessageEvent(event_id="1", content="calculate")
        )
        assert stopped.termination_reason == "turn_token_limit"
        assert not any(e.kind == "tool_call" for e in stopped.events)
        resumed = await runtime.advance(
            session.session_id, MessageEvent(event_id="2", content="continue")
        )
        assert resumed.status == "completed"
        assert len(seen) == 2 and seen[0]["max_tokens"] == 6
        assert any(
            m["role"] == "tool" and "not executed" in m["content"] for m in seen[1]["messages"]
        )
        await backend.aclose()

    asyncio.run(scenario())
