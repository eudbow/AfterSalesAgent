"""框架无关的应用数据结构：HTTP/模型层不接触隐藏评分答案。"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, JsonValue, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RunLimits(Contract):
    max_model_calls: int = Field(default=12, ge=1)
    max_tool_calls: int = Field(default=20, ge=1)
    max_active_seconds: float = Field(default=120, gt=0)
    max_tokens: int = Field(default=24000, ge=1)
    max_turn_model_calls: int = Field(default=6, ge=1)
    max_turn_tool_calls: int = Field(default=10, ge=1)
    max_turn_active_seconds: float = Field(default=30, gt=0)
    max_turn_tokens: int = Field(default=12000, ge=1)


class RunContext(Contract):
    environment_id: str = Field(min_length=1)
    backend: Literal["mock", "deepseek"] = "mock"
    tool_transport: Literal["native", "mcp"] = "native"
    model_name: str | None = None
    limits: RunLimits = Field(default_factory=RunLimits)
    policy_version: str = "tau2-v1.0.1-retail"
    skill_id: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9-]{0,63}$")
    skill_version: str = "none"
    skill_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    skill_snapshot_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    knowledge_version: str = "full-policy"
    knowledge_fingerprint: str | None = None
    temperature: float = Field(default=0, ge=0, le=2)
    max_output_tokens: int = Field(default=1024, ge=1, le=8192)

    @model_validator(mode="after")
    def coherent_skill_selection(self) -> RunContext:
        import re

        if self.skill_id is None:
            if self.skill_version != "none" or self.skill_sha256 or self.skill_snapshot_sha256:
                raise ValueError("No-Skill context must not name a version or hash")
        elif self.skill_version != "released" and not re.fullmatch(
            r"[0-9]+\.[0-9]+\.[0-9]+", self.skill_version
        ):
            raise ValueError("Select a concrete Skill version or released")
        return self


class MessageEvent(Contract):
    kind: Literal["message"] = "message"
    event_id: str = Field(min_length=1)
    content: str = Field(min_length=1)


class ConfirmationEvent(Contract):
    # 确认具体操作，而不是让模型声明“用户已确认”。
    kind: Literal["confirmation"] = "confirmation"
    event_id: str = Field(min_length=1)
    action_id: str = Field(min_length=1)
    approved: bool


UserEvent = Annotated[MessageEvent | ConfirmationEvent, Field(discriminator="kind")]


class ToolSpec(Contract):
    name: str = Field(min_length=1)
    description: str
    arguments_schema: dict[str, JsonValue]
    effect: Literal["read", "write"]


class ToolCall(Contract):
    call_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: dict[str, JsonValue]


class ToolError(Contract):
    code: Literal[
        "invalid_arguments",
        "not_found_or_forbidden",
        "invalid_state",
        "policy_denied",
        "confirmation_required",
        "confirmation_stale",
        "timeout",
        "infrastructure_error",
    ]
    message: str
    retryable: bool = False


class ToolResult(Contract):
    call_id: str
    status: Literal["ok", "error"]
    data: JsonValue = None
    error: ToolError | None = None

    @model_validator(mode="after")
    def consistent_error(self) -> ToolResult:
        if (self.status == "error") != (self.error is not None):
            raise ValueError("error status and error payload must agree")
        return self


class ToolContext(Contract):
    # 这些字段只由服务端构造，不能暴露为模型可填写的工具参数。
    session_id: str
    environment_id: str
    authenticated_user_id: str | None = None
    approved_action_id: str | None = None
    idempotency_key: str | None = None


class PendingAction(Contract):
    policy_basis: list[dict[str, JsonValue]] = Field(default_factory=list)
    action_id: str
    tool_name: str
    arguments: dict[str, JsonValue]
    order_id: str
    order_version: str
    summary: str
    expires_at: AwareDatetime


class SessionView(Contract):
    session_id: str
    environment_id: str
    authenticated_user_id: str | None = None
    status: Literal["ready", "waiting_user", "waiting_confirmation", "closed"] = "ready"
    pending_action: PendingAction | None = None


class TraceEvent(Contract):
    schema_version: Literal["1.0"] = "1.0"
    event_id: str
    session_id: str
    sequence: int = Field(ge=0)
    kind: Literal[
        "user_message",
        "model_request",
        "model_response",
        "model_error",
        "tool_call",
        "tool_result",
        "pending_action",
        "confirmation",
        "state_change",
        "retry",
        "turn_end",
    ]
    payload: dict[str, JsonValue]
    created_at: AwareDatetime


class TurnResult(Contract):
    sources: list[dict[str, JsonValue]] = Field(default_factory=list)
    session: SessionView
    reply: str
    status: Literal[
        "waiting_user",
        "waiting_confirmation",
        "completed",
        "rejected",
        "handoff",
        "failed",
        "truncated",
    ]
    events: list[TraceEvent] = Field(default_factory=list)
    termination_reason: str | None = None


class ModelMessage(Contract):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None


class ModelRequest(Contract):
    messages: list[ModelMessage]
    tools: list[ToolSpec]
    model_name: str | None = None
    temperature: float = Field(default=0, ge=0, le=2)
    max_output_tokens: int = Field(default=1024, ge=1, le=8192)
    tool_choice: Literal["auto", "required", "none"] = "auto"


class ModelResponse(Contract):
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    provider_metadata: dict[str, JsonValue] = Field(default_factory=dict)
    finish_reason: Literal["stop", "tool_calls", "length"] | None = None


class VisibleTaskInput(Contract):
    """只有真实首条用户消息；不是完整用户模拟器指令或答案。"""

    task_id: str
    first_user_message: str
