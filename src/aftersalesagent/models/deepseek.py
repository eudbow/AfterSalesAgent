"""HTTPX适配器：非思考模式、非流式、单次网络请求；默认不允许真实联网。"""

from __future__ import annotations

import json
import time
from typing import Any
from urllib.parse import urlsplit

import httpx
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    SecretStr,
    ValidationError,
    field_validator,
)

from aftersalesagent.models.errors import ModelBackendFault
from aftersalesagent.schemas.contracts import ModelRequest, ModelResponse, ToolCall


class DeepSeekOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    model: str = Field(min_length=1)
    base_url: str = "https://api.deepseek.com"
    timeout_seconds: float = Field(default=25, gt=0, le=120)
    max_output_tokens: int = Field(default=1024, ge=1, le=8192)
    temperature: float = Field(default=0, ge=0, le=2)

    @field_validator("base_url")
    @classmethod
    def official_endpoint(cls, value: str) -> str:
        url = urlsplit(value)
        if (
            url.scheme != "https"
            or url.hostname != "api.deepseek.com"
            or url.port not in (None, 443)
            or url.username
            or url.password
            or url.query
            or url.fragment
            or url.path.rstrip("/") not in ("", "/v1")
        ):
            raise ValueError(
                "Use the official DeepSeek HTTPS endpoint without query or credentials"
            )
        return value.rstrip("/")


def strict_json(value: str):
    def invalid(_value):
        raise ValueError("Nonfinite JSON is not permitted")

    return json.loads(value, parse_constant=invalid)


def encode_request(request: ModelRequest, options: DeepSeekOptions) -> dict:
    if request.model_name and request.model_name != options.model:
        raise ModelBackendFault("model_configuration_mismatch")
    messages: list[dict[str, Any]] = []
    outstanding: set[str] = set()
    for message in request.messages:
        item: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.role == "tool":
            if not message.tool_call_id or message.tool_call_id not in outstanding:
                raise ModelBackendFault("invalid_message_history")
            item["tool_call_id"] = message.tool_call_id
            outstanding.remove(message.tool_call_id)
        else:
            if outstanding:
                raise ModelBackendFault("invalid_message_history")
            if message.tool_calls:
                if message.role != "assistant":
                    raise ModelBackendFault("invalid_message_history")
                ids = [call.call_id for call in message.tool_calls]
                if len(set(ids)) != len(ids):
                    raise ModelBackendFault("invalid_message_history")
                outstanding.update(ids)
                item["content"] = message.content or None
                item["tool_calls"] = [
                    {
                        "id": call.call_id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(
                                call.arguments, ensure_ascii=False, allow_nan=False
                            ),
                        },
                    }
                    for call in message.tool_calls
                ]
        messages.append(item)
    if outstanding:
        raise ModelBackendFault("invalid_message_history")
    body: dict[str, Any] = {
        "model": options.model,
        "messages": messages,
        "stream": False,
        "thinking": {"type": "disabled"},
        "temperature": request.temperature,
        "max_tokens": min(request.max_output_tokens, options.max_output_tokens),
    }
    if request.tools:
        body["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.arguments_schema,
                },
            }
            for tool in request.tools
        ]
        body["tool_choice"] = request.tool_choice
    return body


def token_value(value) -> int | None:
    return value if type(value) is int and value >= 0 else None


def parse_response(payload: dict) -> ModelResponse:
    usage = payload.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    prompt, completion = (
        token_value(usage.get("prompt_tokens")),
        token_value(usage.get("completion_tokens")),
    )
    metadata: dict[str, JsonValue] = {
        key: value
        for key in ("id", "model", "system_fingerprint", "created")
        if isinstance(value := payload.get(key), (str, int)) and not isinstance(value, bool)
    }
    for key in ("prompt_cache_hit_tokens", "prompt_cache_miss_tokens"):
        if (value := token_value(usage.get(key))) is not None:
            metadata[key] = value
    observation = ModelResponse(
        input_tokens=prompt, output_tokens=completion, provider_metadata=metadata
    )
    try:
        choices = payload["choices"]
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError("Expected one choice")
        choice = choices[0]
        reason, message = choice["finish_reason"], choice["message"]
        if message.get("role") != "assistant":
            raise ValueError("Expected assistant role")
        content = message.get("content") or ""
        if not isinstance(content, str):
            raise ValueError("Expected text")
        if reason == "length":
            # 截断的JSON参数不可执行，但已报告usage仍必须计入预算。
            return observation.model_copy(update={"content": content, "finish_reason": "length"})
        if reason in {"content_filter", "insufficient_system_resource", "aborted"}:
            raise ModelBackendFault(
                "model_" + reason, retryable=reason != "content_filter", observation=observation
            )
        if reason not in {"stop", "tool_calls"}:
            raise ValueError("Unknown finish reason")
        calls = []
        raw_calls = message.get("tool_calls") or []
        if not isinstance(raw_calls, list):
            raise ValueError("Expected tool calls")
        for call in raw_calls:
            if call.get("type") != "function":
                raise ValueError("Unsupported tool type")
            function = call["function"]
            args = strict_json(function["arguments"])
            if not isinstance(args, dict):
                raise ValueError("Tool arguments must be an object")
            calls.append(ToolCall(call_id=call["id"], name=function["name"], arguments=args))
        if len({call.call_id for call in calls}) != len(calls):
            raise ValueError("Duplicate tool IDs")
        if (reason == "tool_calls") != bool(calls) or (not calls and not content.strip()):
            raise ValueError("Finish reason does not match response")
        return ModelResponse(
            content=content,
            tool_calls=calls,
            input_tokens=prompt,
            output_tokens=completion,
            finish_reason=reason,
            provider_metadata=metadata,
        )
    except ModelBackendFault:
        raise
    except (KeyError, TypeError, ValueError, ValidationError):
        raise ModelBackendFault("model_protocol_error", observation=observation) from None


class DeepSeekBackend:
    def __init__(
        self,
        options: DeepSeekOptions,
        api_key: SecretStr,
        *,
        allow_network: bool = False,
        transport: httpx.MockTransport | None = None,
    ):
        if transport is None and not allow_network:
            raise ValueError("Real API access requires explicit --enable-paid-api")
        if not api_key.get_secret_value().strip():
            raise ValueError("Configure DEEPSEEK_API_KEY locally before live requests")
        self.options = options
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(options.timeout_seconds),
            transport=transport,
            headers={"Authorization": "Bearer " + api_key.get_secret_value()},
            follow_redirects=False,
            trust_env=False,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def generate(self, request: ModelRequest) -> ModelResponse:
        body = encode_request(request, self.options)
        started = time.monotonic()
        try:
            response = await self._client.post(
                self.options.base_url + "/chat/completions", json=body
            )
        except httpx.TimeoutException:
            raise ModelBackendFault("model_api_timeout", retryable=True) from None
        except httpx.RequestError:
            raise ModelBackendFault("model_connection_error", retryable=True) from None
        status = response.status_code
        if status != 200:
            code = {
                400: "model_invalid_request",
                401: "model_authentication_failed",
                402: "model_insufficient_balance",
                403: "model_access_denied",
                422: "model_invalid_request",
                429: "model_rate_limited",
            }.get(status, "model_server_error" if status >= 500 else "model_http_error")
            raise ModelBackendFault(
                code, retryable=status == 429 or status >= 500, status_code=status
            )
        try:
            payload = strict_json(response.text)
            if not isinstance(payload, dict):
                raise ValueError("Expected object")
        except (ValueError, TypeError):
            raise ModelBackendFault("model_protocol_error") from None
        result = parse_response(payload)
        return result.model_copy(
            update={
                "provider_metadata": {
                    **result.provider_metadata,
                    "latency_seconds": time.monotonic() - started,
                    "thinking": "disabled",
                    "temperature": request.temperature,
                    "requested_max_tokens": body["max_tokens"],
                }
            }
        )
