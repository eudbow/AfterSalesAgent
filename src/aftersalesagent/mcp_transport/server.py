"""独立 MCP Server。stdout 只供协议使用，业务上下文不进入模型工具参数。"""

import argparse
import asyncio
import json
import os
import secrets
from pathlib import Path

from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool, ToolAnnotations
from pydantic import Field, ValidationError

from aftersalesagent.business.database import RetailStore
from aftersalesagent.business.service import RetailService
from aftersalesagent.knowledge.corpus import load_corpus
from aftersalesagent.knowledge.e4 import E4Config, E4Index, build_index
from aftersalesagent.knowledge.retail import (
    RETAIL_SEARCH_SPEC,
    NativeRetailPolicyTool,
    RetailPolicyIndex,
)
from aftersalesagent.knowledge.retrieval import SQLiteKnowledgeIndex
from aftersalesagent.schemas.contracts import Contract, ToolCall, ToolContext, ToolResult
from aftersalesagent.tools.registry import NativeRetailTool

ROOT = Path(__file__).resolve().parents[3]


class SearchPolicy(Contract):
    query: str = Field(min_length=1, max_length=2000)
    filters: dict[str, str] = Field(default_factory=dict)
    top_k: int = Field(default=5, ge=1, le=30)


class SelectPolicy(Contract):
    filters: dict[str, str] = Field(default_factory=dict)


def packed(data: dict, *, error=False) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(data, ensure_ascii=False))],
        structured_content=data,
        is_error=error,
    )


def fault(call_id: str, code: str, message: str) -> ToolResult:
    return ToolResult.model_validate(
        {"call_id": call_id, "status": "error", "error": {"code": code, "message": message}}
    )


async def serve(args):
    if args.knowledge_config:
        config_path = args.knowledge_config.resolve()
        if not config_path.is_relative_to(ROOT):
            raise ValueError("Knowledge configuration escapes project")
        config = E4Config.model_validate_json(config_path.read_text("utf-8"))
        index = build_index(ROOT, config)
    else:
        chunks, _ = load_corpus(ROOT)
        index = SQLiteKnowledgeIndex(chunks)
    store = RetailStore(args.database) if args.database else None
    service = RetailService(store) if store else None
    specs = (
        {s.name: s for s in service.specs() if args.allow_proposals or s.effect == "read"}
        if service
        else {}
    )
    retail_index = (
        RetailPolicyIndex(method=args.retail_retrieval)
        if service and args.retail_rag and not args.retail_rag_v2 else None
    )
    if service and args.retail_rag_v2:
        if retail_index is not None:
            retail_index.close()
        from aftersalesagent.knowledge.retail_v2 import RetailV2Index

        retail_index = RetailV2Index()
    if retail_index is not None:
        service.policy_basis = retail_index.action_basis
        specs[RETAIL_SEARCH_SPEC.name] = RETAIL_SEARCH_SPEC
    tools = [
        Tool(
            name=s.name,
            description=s.description,
            input_schema=s.arguments_schema,
            annotations=ToolAnnotations(read_only_hint=s.effect == "read"),
            _meta={"aftersales/effect": s.effect},
        )
        for s in specs.values()
    ]
    tools += [
        Tool(
            name="search_policy",
            description="Retrieve versioned after-sales evidence; no model call.",
            input_schema=SearchPolicy.model_json_schema(),
            annotations=ToolAnnotations(read_only_hint=True),
        ),
        Tool(
            name="get_applicable_policies",
            description="Read all eligible curated policy cards.",
            input_schema=SelectPolicy.model_json_schema(),
            annotations=ToolAnnotations(read_only_hint=True),
        ),
    ]
    capability = os.environ.get("AFTERSALES_MCP_CAPABILITY", "")

    async def listing(_context, _params):
        return ListToolsResult(tools=tools)

    async def calling(_context, params):
        meta = params.meta or {}
        call_id = str(meta.get("aftersales/call_id", "mcp-call"))
        arguments = params.arguments or {}
        try:
            if params.name == "search_policy":
                query = SearchPolicy.model_validate(arguments)
                ranked = index.ranked(query.query, query.filters, query.top_k)
                return packed(
                    {
                        "evidence": [
                            {"chunk": c.model_dump(mode="json"), "score": score}
                            for c, score in ranked
                        ],
                        **(
                            {
                                "index_fingerprint": index.index_fingerprint,
                                "retrieval_trace": index.last_trace,
                            }
                            if isinstance(index, E4Index)
                            else {}
                        ),
                    }
                )
            if params.name == "get_applicable_policies":
                query = SelectPolicy.model_validate(arguments)
                return packed(
                    {
                        "evidence": [
                            c.model_dump(mode="json") for c in index.select(query.filters)
                        ],
                        **(
                            {"index_fingerprint": index.index_fingerprint}
                            if isinstance(index, E4Index)
                            else {}
                        ),
                    }
                )
            if params.name not in specs or service is None:
                result = fault(call_id, "invalid_arguments", "Unknown or disabled tool")
            elif not capability or not secrets.compare_digest(
                str(meta.get("aftersales/capability", "")), capability
            ):
                result = fault(call_id, "policy_denied", "Missing trusted MCP capability")
            else:
                context = ToolContext.model_validate(meta.get("aftersales/context"))
                if (
                    context.session_id != args.session_id
                    or context.environment_id != service.environment_id
                ):
                    result = fault(call_id, "policy_denied", "MCP session binding mismatch")
                elif specs[params.name].effect == "write" and not context.idempotency_key:
                    result = fault(
                        call_id, "policy_denied", "Stateful calls require an operation key"
                    )
                else:
                    call = ToolCall(call_id=call_id, name=params.name, arguments=arguments)
                    tool = (
                        NativeRetailPolicyTool(service, retail_index)
                        if params.name == RETAIL_SEARCH_SPEC.name
                        else NativeRetailTool(service, specs[params.name])
                    )
                    result = await tool.invoke(call, context)
                    # 只允许启动者在隔离数据上注入故障，不是模型可调用的参数。
                    if (
                        args.fault_after_commit
                        and specs[params.name].effect == "write"
                        and result.status == "ok"
                        and not args.fault_after_commit.exists()
                    ):
                        args.fault_after_commit.parent.mkdir(parents=True, exist_ok=True)
                        args.fault_after_commit.write_text("committed; response lost", "utf-8")
                        if args.fault_delay_seconds:
                            await asyncio.sleep(args.fault_delay_seconds)
                        else:
                            os._exit(73)
            return packed(result.model_dump(mode="json"), error=result.status == "error")
        except (ValidationError, ValueError):
            result = fault(call_id, "invalid_arguments", "Invalid tool arguments or metadata")
            return packed(result.model_dump(mode="json"), error=True)

    server = Server("AfterSalesAgent", version="e2-v1", on_list_tools=listing, on_call_tool=calling)
    try:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())
    finally:
        index.close()
        if retail_index:
            retail_index.close()
        if store:
            store.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retail-rag", action="store_true")
    parser.add_argument("--retail-rag-v2", action="store_true")
    parser.add_argument("--retail-retrieval", choices=["bm25", "dense", "hybrid"], default="bm25")
    parser.add_argument("--database", type=Path)
    parser.add_argument("--knowledge-config", type=Path)
    parser.add_argument("--session-id")
    parser.add_argument("--allow-proposals", action="store_true")
    parser.add_argument("--fault-after-commit", type=Path)
    parser.add_argument("--fault-delay-seconds", type=float, default=0)
    args = parser.parse_args()
    if bool(args.database) != bool(args.session_id):
        parser.error("Database and trusted session binding must be supplied together")
    if args.database and not args.database.is_file():
        parser.error("MCP only opens an existing application-owned database")
    asyncio.run(serve(args))


if __name__ == "__main__":
    main()
