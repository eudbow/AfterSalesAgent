"""MCP结果适配回原工具契约；不在适配器重新实现业务规则。"""

from pathlib import Path

from aftersalesagent.knowledge.corpus import KnowledgeChunk, eligible
from aftersalesagent.mcp_transport.client import MCPConnection
from aftersalesagent.schemas.contracts import ToolError, ToolResult, ToolSpec


class MCPRetailTool:
    def __init__(self, owner, spec: ToolSpec):
        self.owner, self._spec = owner, spec

    @property
    def spec(self):
        return self._spec

    async def invoke(self, call, context):
        connection = self.owner.connection(context.session_id)
        try:
            await connection.connect()
            discovered = next((t for t in connection.discovery if t.name == call.name), None)
            if (
                discovered is None
                or discovered.input_schema != self.spec.arguments_schema
                or discovered.description != self.spec.description
                or (discovered.meta or {}).get("aftersales/effect") != self.spec.effect
            ):
                raise ValueError("MCP contract drift")
            result = ToolResult.model_validate(
                await connection.call(
                    call.name, call.arguments, context=context, call_id=call.call_id
                )
            )
            if result.call_id != call.call_id:
                raise ValueError("MCP call correlation mismatch")
            return result
        except (ConnectionError, TimeoutError):
            # 抛给原 Harness 重试机制，仍沿用原业务幂等键。
            raise
        except ValueError:
            return ToolResult(
                call_id=call.call_id,
                status="error",
                error=ToolError(code="infrastructure_error", message="MCP contract mismatch"),
            )


class MCPRetailProvider:
    def __init__(self, service, *, trace_path=None, fault_after_commit=None):
        self.service, self.trace_path = service, trace_path
        self.fault_after_commit = fault_after_commit
        self._connection: MCPConnection | None = None

    def connection(self, sid):
        if self._connection is None:
            self._connection = MCPConnection(
                database=self.service.store.path,
                session_id=sid,
                allow_proposals=True,
                trace_path=self.trace_path,
                fault_after_commit=self.fault_after_commit,
            )
        return self._connection

    async def aclose(self):
        if self._connection:
            await self._connection.aclose()


class MCPKnowledgeProvider:
    def __init__(
        self, index, *, trace_path: Path | None = None, knowledge_config: Path | None = None
    ):
        self.index = index
        self.tokenizer = index.tokenizer
        if hasattr(index, "connection"):
            self.connection = (
                index.connection
            )  # E1 compatibility; E4 supplies an independent ledger.
        timeout = index.config.worker_timeout_seconds + 15 if knowledge_config else 15
        self.remote = MCPConnection(
            trace_path=trace_path, knowledge_config=knowledge_config, timeout=timeout
        )
        self.knowledge_config = knowledge_config
        self.last_trace = None

    def select(self, filters):
        # 只用于原账本/完整性信息，回答上下文的实际供给走远端。
        return self.index.select(filters)

    async def provide(self, query, filters, top_k, *, full=False):
        name = "get_applicable_policies" if full else "search_policy"
        arguments = (
            {"filters": filters} if full else {"query": query, "filters": filters, "top_k": top_k}
        )
        result = await self.remote.call(name, arguments)
        if (
            self.knowledge_config
            and result.get("index_fingerprint") != self.index.index_fingerprint
        ):
            raise ValueError("MCP retrieval configuration mismatch")
        if "evidence" not in result:
            raise ValueError("MCP evidence unavailable")
        records = result["evidence"]
        chunks = [KnowledgeChunk.model_validate(c if full else c["chunk"]) for c in records]
        if len({c.chunk_id for c in chunks}) != len(chunks) or (not full and len(chunks) > top_k):
            raise ValueError("MCP duplicate or oversized evidence")
        # 验证远端证据确实来自本版固定语料，且没有越过请求范围。
        allowed = {c.chunk_id: c for c in eligible(self.index.chunks, filters)}
        if any(c.chunk_id not in allowed or c != allowed[c.chunk_id] for c in chunks):
            raise ValueError("MCP evidence provenance mismatch")
        self.last_trace = result.get("retrieval_trace")
        return chunks

    async def aclose(self):
        await self.remote.aclose()

    def close(self):
        self.index.close()
