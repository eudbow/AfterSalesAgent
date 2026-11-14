"""评测的私有边界：不可作为模型请求或 Runtime 状态字段。"""

from typing import Protocol

from pydantic import JsonValue

from aftersalesagent.schemas.contracts import Contract, TraceEvent


class VerificationResult(Contract):
    success: bool
    checks: dict[str, bool]
    failure_labels: list[str]


class Evaluator(Protocol):
    async def evaluate(
        self,
        private_criteria: dict[str, JsonValue],
        initial_state: dict[str, JsonValue],
        final_state: dict[str, JsonValue],
        events: list[TraceEvent],
    ) -> VerificationResult: ...
