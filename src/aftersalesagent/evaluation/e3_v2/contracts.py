"""显式用户接口与逐步公开的事实；模拟器不能从未来期望参数补答案。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Protocol

from pydantic import Field, JsonValue, model_validator

from aftersalesagent.schemas.contracts import Contract, TurnResult, UserEvent

Language = Literal["zh", "en"]
Outcome = Literal[
    "running", "completed", "agent_failure", "coverage_gap", "interaction_limit", "runtime_failure"
]
ToolName = Literal[
    "cancel_pending_order", "modify_pending_order_address", "return_delivered_order_items"
]
Slot = Literal["reason", "zip", "items", "payment", "email", "order_id", "address"]
SLOT_FIELD = {
    "reason": "reason",
    "zip": "zip",
    "items": "item_ids",
    "payment": "payment_method_id",
    "order_id": "order_id",
    "email": "email",
}
ADDRESS_FIELDS = ("address1", "address2", "city", "state", "country", "zip")


def normalise(arguments: dict) -> dict:
    output = dict(arguments)
    if "item_ids" in output:
        output["item_ids"] = sorted(output["item_ids"])
    return output


class Intent(Contract):
    tool_name: ToolName
    arguments: dict[str, JsonValue]


class FactUpdate(Contract):
    tool_name: ToolName | None = None
    arguments: dict[str, JsonValue] = Field(default_factory=dict)


def apply_update(current: Intent, update: FactUpdate) -> Intent:
    name = update.tool_name or current.tool_name
    # 切换办理意图时不把旧地址/商品等字段带入另一种工具。
    args = (
        dict(current.arguments)
        if name == current.tool_name
        else {"order_id": current.arguments["order_id"]}
    )
    args.update(update.arguments)
    return Intent(tool_name=name, arguments=args)


class Stage(Contract):
    trigger: Literal["proposal", "question", "refusal"]
    label: str = Field(min_length=1)
    expected: Intent | None = None
    slot: Slot | None = None
    reply: str = Field(min_length=1)
    reply_language: Language
    update: FactUpdate

    @model_validator(mode="after")
    def coherent_trigger(self) -> Stage:
        if (self.trigger == "proposal") != (self.expected is not None):
            raise ValueError("Only proposal stages have an expected card")
        if (self.trigger == "question") != (self.slot is not None):
            raise ValueError("Only question stages declare a missing slot")
        return self


class UserScript(Contract):
    task_id: str
    source_user_id: str
    source_task: str
    family: str
    title: str
    initial_language: Language
    known_identity: dict[str, str]
    initial_intent: Intent
    stages: tuple[Stage, ...]
    final_intent: Intent
    max_events: int = Field(default=12, ge=2, le=16)

    @model_validator(mode="after")
    def progressive_facts(self) -> UserScript:
        state = self.initial_intent
        if set(self.known_identity) != {"email"}:
            raise ValueError("Only the already disclosed email is an identity answer source")
        for stage in self.stages:
            if stage.expected is not None and (
                stage.expected.tool_name != state.tool_name
                or normalise(stage.expected.arguments) != normalise(state.arguments)
            ):
                raise ValueError("A card cannot require facts not yet disclosed")
            if stage.slot in SLOT_FIELD:
                field = SLOT_FIELD[stage.slot]
                if field in state.arguments or field in self.known_identity:
                    raise ValueError("A required question must concern an unrevealed field")
            state = apply_update(state, stage.update)
        if state.tool_name != self.final_intent.tool_name or normalise(
            state.arguments
        ) != normalise(self.final_intent.arguments):
            raise ValueError("Final intention must be fully disclosed by user messages")
        return self


class SimulatorOptions(Contract):
    per_slot_per_stage: int = Field(default=1, ge=0, le=2)
    per_stage: int = Field(default=2, ge=0, le=4)
    total_optional_replies: int = Field(default=4, ge=0, le=8)


class SimulatorState(Contract):
    step_index: int
    outcome: Outcome
    code: str | None = None
    language: Language
    known_intent: Intent
    actions: list[dict] = Field(default_factory=list)
    issues: list[dict] = Field(default_factory=list)
    optional_replies: int = 0


class UserSimulator(Protocol):
    def describe(self) -> dict: ...
    def next_event(self, result: TurnResult) -> UserEvent | None: ...
    def state(self) -> SimulatorState: ...
    def on_limit(self) -> None: ...


UserFactory = Callable[[UserScript], UserSimulator]
