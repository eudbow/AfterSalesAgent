"""模型后端不执行工具；厂商特殊字段由适配器封装。"""

from typing import Protocol

from aftersalesagent.schemas.contracts import ModelRequest, ModelResponse


class ModelBackend(Protocol):
    async def generate(self, request: ModelRequest) -> ModelResponse: ...
