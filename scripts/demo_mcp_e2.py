"""独立SDK客户端：不用Runtime/适配器/Key，真实stdio发现工具并查询订单与政策。"""

import asyncio
import json
import uuid
from pathlib import Path

from mcp import Client

from aftersalesagent.business.database import clone_database, import_pinned_database
from aftersalesagent.business.service import RetailService
from aftersalesagent.mcp_transport.client import MCPConnection
from aftersalesagent.models.scenarios import SCENARIOS
from aftersalesagent.schemas.contracts import RunContext, ToolContext

ROOT = Path(__file__).resolve().parents[1]


async def main():
    folder = ROOT / ".local/e2_demos" / uuid.uuid4().hex[:10]
    folder.mkdir(parents=True)
    template = ROOT / ".local/databases/retail_template.sqlite"
    if not template.exists():
        import_pinned_database(ROOT, template).close()
    store = clone_database(template, folder / "episode.sqlite")
    service = RetailService(store)
    sid = service.start(RunContext(environment_id=service.environment_id)).session_id
    launch = MCPConnection(database=store.path, session_id=sid)
    rows = []
    try:
        async with Client(launch.parameters()) as client:
            tools = (await client.list_tools()).tools
            assert "get_order_details" in [t.name for t in tools]
            assert "cancel_pending_order" not in [t.name for t in tools]
            for i, (name, arguments) in enumerate(SCENARIOS["query"]["steps"]):
                view = service.view(sid)
                context = ToolContext(
                    session_id=sid,
                    environment_id=service.environment_id,
                    authenticated_user_id=view.authenticated_user_id,
                    idempotency_key=f"demo:{i}",
                )
                result = await client.call_tool(
                    name,
                    arguments,
                    meta={
                        "aftersales/context": context.model_dump(mode="json"),
                        "aftersales/capability": launch.capability,
                        "aftersales/call_id": f"query-{i}",
                    },
                )
                assert not result.is_error, result
                rows.append(
                    {"tool": name, "arguments": arguments, "result": result.structured_content}
                )
            result = await client.call_tool(
                "search_policy",
                {"query": "银行卡退货凭证材料", "filters": {"scope": "ikea_cn"}, "top_k": 5},
            )
            assert not result.is_error
            assert any(
                c["chunk"]["chunk_id"] == "R02" for c in result.structured_content["evidence"]
            )
            rows.append({"tool": "search_policy", "result": result.structured_content})
            report = {
                "passed": True,
                "mode": "independent_sdk_stdio_client",
                "protocol_version": client.protocol_version,
                "tools": [t.model_dump(mode="json", by_alias=True) for t in tools],
                "calls": rows,
                "real_model_calls": 0,
                "gpu_experiments": 0,
            }
    finally:
        store.close()
    target = folder / "result.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", "utf-8")
    print(
        json.dumps(
            {
                "passed": True,
                "report": str(target),
                "tools": len(report["tools"]),
                "protocol_version": report["protocol_version"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
