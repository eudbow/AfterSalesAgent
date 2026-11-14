"""Runtime sees a registry; tool names and business behavior stay outside Runtime."""

from aftersalesagent.business.service import RetailService
from aftersalesagent.harness.execution import run_sync
from aftersalesagent.schemas.contracts import ToolCall, ToolContext, ToolError, ToolResult, ToolSpec


class NativeRetailTool:
    def __init__(self, service: RetailService, spec: ToolSpec):
        self.service, self._spec = service, spec

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    async def invoke(self, call: ToolCall, context: ToolContext) -> ToolResult:
        view = await run_sync(lambda: self.service.view(context.session_id))
        if (
            context.environment_id != view.environment_id
            or context.authenticated_user_id != view.authenticated_user_id
        ):
            return ToolResult(
                call_id=call.call_id,
                status="error",
                error=ToolError(
                    code="policy_denied", message="Tool context does not match server session"
                ),
            )
        return await run_sync(
            lambda: self.service.dispatch(
                context.session_id, call, idempotency_key=context.idempotency_key
            )
        )


class ToolRegistry:
    def __init__(self, service: RetailService, *, provider=None):
        self.service, self.provider = service, provider
        self.transport = "mcp" if provider is not None else "native"
        if provider is None:
            self._tools = {s.name: NativeRetailTool(service, s) for s in service.specs()}
        else:
            from aftersalesagent.mcp_transport.adapters import MCPRetailTool

            self._tools = {s.name: MCPRetailTool(provider, s) for s in service.specs()}

    async def aclose(self):
        if self.provider is not None:
            await self.provider.aclose()

    def specs(self) -> list[ToolSpec]:
        return [tool.spec for tool in self._tools.values()]

    async def invoke(self, call: ToolCall, context: ToolContext) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(
                call_id=call.call_id,
                status="error",
                error=ToolError(code="invalid_arguments", message="Unknown tool"),
            )
        return await tool.invoke(call, context)
