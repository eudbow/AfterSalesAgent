"""稳定的 Runtime 契约；API 不传入 LangGraph State 或隐藏评分材料。"""

from typing import Protocol

from aftersalesagent.schemas.contracts import RunContext, SessionView, TurnResult, UserEvent


class AgentRuntime(Protocol):
    async def start(self, context: RunContext) -> SessionView: ...
    async def advance(self, session_id: str, user_event: UserEvent) -> TurnResult: ...
    async def get_state(self, session_id: str) -> SessionView: ...
