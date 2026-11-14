"""Policy 只选择下一步，业务执行由工具/环境负责。"""

from typing import Protocol

from aftersalesagent.schemas.contracts import ModelRequest, ModelResponse


class DecisionPolicy(Protocol):
    async def decide(self, request: ModelRequest) -> ModelResponse: ...
