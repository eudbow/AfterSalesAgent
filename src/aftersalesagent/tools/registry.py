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
    def __init__(
        self,
        service: RetailService,
        *,
        provider=None,
        retail_index=None,
        include_retail_search=True,
    ):
        self.service, self.provider = service, provider
        self.retail_index = retail_index
        if retail_index is not None:
            service.policy_basis = retail_index.action_basis
        self.transport = "mcp" if provider is not None else "native"
        if provider is None:
            self._tools = {s.name: NativeRetailTool(service, s) for s in service.specs()}
        else:
            from aftersalesagent.mcp_transport.adapters import MCPRetailTool

            self._tools = {s.name: MCPRetailTool(provider, s) for s in service.specs()}

        if retail_index is not None and include_retail_search:
            from aftersalesagent.knowledge.retail import RETAIL_SEARCH_SPEC, NativeRetailPolicyTool

            if provider is None:
                tool = NativeRetailPolicyTool(service, retail_index)
            else:
                tool = MCPRetailTool(provider, RETAIL_SEARCH_SPEC)
            self._tools[RETAIL_SEARCH_SPEC.name] = tool

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
        result = await tool.invoke(call, context)
        if (
            self.retail_index is not None
            and call.name == "search_retail_policy"
            and result.status == "ok"
        ):
            try:
                self.retail_index.validate_result(
                    result.data,
                    session_id=context.session_id,
                    call_id=call.call_id,
                    arguments=call.arguments,
                )
            except ValueError:
                return ToolResult(
                    call_id=call.call_id,
                    status="error",
                    error=ToolError(
                        code="infrastructure_error", message="Retail evidence verification failed"
                    ),
                )
        return result
