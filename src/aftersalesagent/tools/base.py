"""原生工具与未来 MCP Adapter 共用接口。"""

from typing import Protocol

from aftersalesagent.schemas.contracts import ToolCall, ToolContext, ToolResult, ToolSpec


class BusinessTool(Protocol):
    @property
    def spec(self) -> ToolSpec: ...
    async def invoke(self, call: ToolCall, context: ToolContext) -> ToolResult: ...
