"""规则、记录边界；具体确认/预算/恢复在 M1/M2 实现。"""

from typing import Protocol

from aftersalesagent.schemas.contracts import ToolCall, ToolContext, ToolError, TraceEvent


class ExecutionGuard(Protocol):
    async def check(self, call: ToolCall, context: ToolContext) -> ToolError | None: ...


class TrajectoryRecorder(Protocol):
    async def append(self, event: TraceEvent) -> None: ...
