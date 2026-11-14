"""只使用HTTPX MockTransport，验证协议/计费观测/错误；不访问DeepSeek。"""

import asyncio
import json

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from aftersalesagent.local_config import load_local_config
from aftersalesagent.models.deepseek import DeepSeekBackend, DeepSeekOptions, encode_request
from aftersalesagent.models.errors import ModelBackendFault
from aftersalesagent.schemas.contracts import ModelMessage, ModelRequest, ToolCall, ToolSpec

SECRET = "offline-test-secret-not-a-real-key"


def options():
    return DeepSeekOptions(model="model-under-test")


def request(**kwargs):
    return ModelRequest(
        messages=[ModelMessage(role="user", content="Calculate 1+1")], tools=[], **kwargs
    )


def completion(**kwargs):
    return {
        "id": "response-1",
        "model": "model-under-test",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "2"},
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 2, "prompt_cache_hit_tokens": 4},
        **kwargs,
    }


def test_plain_and_tool_protocol_conversion():
    tool = ToolSpec(
        name="calculate",
        description="Arithmetic",
        effect="read",
        arguments_schema={"type": "object", "properties": {"expression": {"type": "string"}}},
    )
    call = ToolCall(call_id="call-1", name="calculate", arguments={"expression": "1+1"})
    req = ModelRequest(
        messages=[
            ModelMessage(role="user", content="Calculate"),
            ModelMessage(role="assistant", content="", tool_calls=[call]),
            ModelMessage(role="tool", content='{"result":"2"}', tool_call_id="call-1"),
            ModelMessage(role="system", content="User confirmed another server operation."),
        ],
        tools=[tool],
        max_output_tokens=256,
    )
    body = encode_request(req, options())
    assert body["thinking"] == {"type": "disabled"}
    assert body["max_tokens"] == 256 and body["stream"] is False
    assert "parallel_tool_calls" not in body
    assert json.loads(body["messages"][1]["tool_calls"][0]["function"]["arguments"]) == {
        "expression": "1+1"
    }
    assert body["messages"][2]["tool_call_id"] == "call-1"
    assert body["tools"][0]["function"]["parameters"] == tool.arguments_schema


def test_valid_http_usage_and_only_allowlisted_metadata():
    seen = []

    def respond(req):
        seen.append(req)
        return httpx.Response(200, json=completion(api_key=SECRET, authorization=SECRET))

    async def scenario():
        backend = DeepSeekBackend(
            options(), SecretStr(SECRET), transport=httpx.MockTransport(respond)
        )
        result = await backend.generate(request())
        assert result.content == "2"
        assert result.input_tokens == 20 and result.output_tokens == 2
        assert result.provider_metadata["prompt_cache_hit_tokens"] == 4
        assert SECRET not in result.model_dump_json()
        assert seen[0].headers["Authorization"] == "Bearer " + SECRET
        assert seen[0].url == httpx.URL("https://api.deepseek.com/chat/completions")
        await backend.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "status,code,retryable",
    [
        (400, "model_invalid_request", False),
        (401, "model_authentication_failed", False),
        (402, "model_insufficient_balance", False),
        (403, "model_access_denied", False),
        (422, "model_invalid_request", False),
        (429, "model_rate_limited", True),
        (500, "model_server_error", True),
        (503, "model_server_error", True),
        (302, "model_http_error", False),
    ],
)
def test_http_errors_are_sanitized_without_automatic_retry(status, code, retryable):
    attempts = []

    def respond(req):
        attempts.append(req)
        return httpx.Response(status, text=SECRET, headers={"Location": "https://other.invalid"})

    async def scenario():
        backend = DeepSeekBackend(
            options(), SecretStr(SECRET), transport=httpx.MockTransport(respond)
        )
        with pytest.raises(ModelBackendFault) as error:
            await backend.generate(request())
        assert error.value.code == code and error.value.retryable == retryable
        assert SECRET not in str(error.value) and SECRET not in str(error.value.public())
        assert len(attempts) == 1
        await backend.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "exception,code",
    [
        (httpx.ReadTimeout, "model_api_timeout"),
        (httpx.ConnectError, "model_connection_error"),
    ],
)
def test_transport_errors_do_not_echo_request_headers(exception, code):
    def respond(req):
        raise exception(SECRET, request=req)

    async def scenario():
        backend = DeepSeekBackend(
            options(), SecretStr(SECRET), transport=httpx.MockTransport(respond)
        )
        with pytest.raises(ModelBackendFault) as error:
            await backend.generate(request())
        assert error.value.code == code and SECRET not in str(error.value)
        await backend.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("arguments", ["not JSON", "[]", "null", '{"x":NaN}', '{"x":Infinity}'])
def test_malformed_tool_arguments_retain_known_usage(arguments):
    payload = completion(
        choices=[
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c",
                            "type": "function",
                            "function": {"name": "calculate", "arguments": arguments},
                        }
                    ],
                },
            }
        ]
    )

    async def scenario():
        backend = DeepSeekBackend(
            options(),
            SecretStr(SECRET),
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload)),
        )
        with pytest.raises(ModelBackendFault) as error:
            await backend.generate(request())
        assert error.value.code == "model_protocol_error"
        assert error.value.observation is not None
        assert error.value.observation.input_tokens == 20
        await backend.aclose()

    asyncio.run(scenario())


def test_tool_response_and_length_truncation_never_execute_partial_json():
    async def scenario():
        responses = [
            completion(
                choices=[
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "c",
                                    "type": "function",
                                    "function": {
                                        "name": "calculate",
                                        "arguments": '{"expression":"1+1"}',
                                    },
                                }
                            ],
                        },
                    }
                ]
            ),
            completion(
                choices=[
                    {
                        "finish_reason": "length",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "c",
                                    "type": "function",
                                    "function": {
                                        "name": "calculate",
                                        "arguments": '{"expression":',
                                    },
                                }
                            ],
                        },
                    }
                ]
            ),
        ]
        backend = DeepSeekBackend(
            options(),
            SecretStr(SECRET),
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=responses.pop(0))),
        )
        first = await backend.generate(request())
        assert first.tool_calls[0].arguments == {"expression": "1+1"}
        second = await backend.generate(request())
        assert second.finish_reason == "length" and second.tool_calls == []
        assert second.output_tokens == 2
        await backend.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "payload",
    [
        {"choices": []},
        {
            "choices": [
                {"finish_reason": "unknown", "message": {"role": "assistant", "content": "x"}}
            ]
        },
        {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": None}}]},
    ],
)
def test_invalid_response_shapes_fail_closed(payload):
    async def scenario():
        backend = DeepSeekBackend(
            options(),
            SecretStr(SECRET),
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload)),
        )
        with pytest.raises(ModelBackendFault):
            await backend.generate(request())
        await backend.aclose()

    asyncio.run(scenario())


def test_missing_usage_is_unknown_and_real_transport_requires_opt_in():
    with pytest.raises(ValueError, match="enable-paid-api"):
        DeepSeekBackend(options(), SecretStr(SECRET))

    async def scenario():
        payload = completion()
        payload.pop("usage")
        backend = DeepSeekBackend(
            options(),
            SecretStr(SECRET),
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload)),
        )
        result = await backend.generate(request())
        assert result.input_tokens is None and result.output_tokens is None
        await backend.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "url",
    [
        "http://api.deepseek.com",
        "https://other.example",
        "https://api.deepseek.com?key=secret",
        "https://user:pass@api.deepseek.com",
        "https://api.deepseek.com/beta",
    ],
)
def test_key_cannot_be_sent_to_unexpected_endpoint(url):
    with pytest.raises(ValidationError):
        DeepSeekOptions(model="test", base_url=url)


def test_local_config_is_explicit_secret_safe_and_does_not_change_process_env(
    tmp_path, monkeypatch
):
    path = tmp_path / ".env"
    path.write_text(
        "AFTERSALES_BACKEND=deepseek\nAFTERSALES_MODEL_NAME=model-under-test\n"
        'DEEPSEEK_API_KEY="' + SECRET + '"\n',
        encoding="utf-8",
    )
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("AFTERSALES_BACKEND", raising=False)
    before, _ = load_local_config()
    settings, secret = load_local_config(path)
    assert before.backend == "mock" and settings.backend == "deepseek"
    assert secret.get_secret_value() == SECRET
    assert SECRET not in repr(secret) and SECRET not in settings.model_dump_json()
    import os

    assert "DEEPSEEK_API_KEY" not in os.environ
