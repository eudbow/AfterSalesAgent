"""Versioned capabilities and final-context dependencies, without model API."""

import asyncio

import pytest

from aftersalesagent.business.service import RetailService
from aftersalesagent.knowledge.retail import RetailSearchArgs
from aftersalesagent.knowledge.retail_v2 import RetailV2Index
from aftersalesagent.mcp_transport.adapters import MCPRetailProvider
from aftersalesagent.models.mock import MockBackend
from aftersalesagent.policies.retail_v2 import (
    answer_issue,
    cancellation_reason_supported,
    validate_tools,
)
from aftersalesagent.runtime.langgraph_runtime import LangGraphRuntime
from aftersalesagent.schemas.contracts import (
    MessageEvent,
    ModelResponse,
    RunContext,
    ToolCall,
    ToolContext,
)
from aftersalesagent.tools.registry import ToolRegistry


def test_capabilities_match_business_tools(store):
    specs = RetailService(store).specs()
    validate_tools(specs)
    with pytest.raises(ValueError, match="mismatch"):
        validate_tools([spec for spec in specs if spec.name != "cancel_pending_order"])


def test_cancellation_reason_requires_user_support():
    assert not cancellation_reason_supported("no longer needed", "想得到赔偿所以取消")
    assert cancellation_reason_supported("no longer needed", "取消原因是我不再需要了")
    assert cancellation_reason_supported("ordered by mistake", "这个订单下错了")
    assert not cancellation_reason_supported("other", "这个订单下错了")


@pytest.mark.parametrize(
    "text,invalid",
    [
        ("只有 pending 订单才能取消。[cite:APP-CAP-V2]", True),
        ("只有 pending 订单才能取消。[cite:POL-07]", False),
        ("建议联系承运方改地址。", True),
        ("退款已到账。", True),
        ("退款记录不能证明已到账。", False),
        ("人工会跟进此事。", True),
        ("仅记录模拟人工交接请求，不代表真人接通。", False),
    ],
)
def test_known_answer_claims(text, invalid):
    assert bool(answer_issue(text)) is invalid


def test_v2_web_skill_and_checkpoint_restore(tmp_path, retail_template):
    from aftersalesagent.api.app import create_app
    from aftersalesagent.config import Settings
    from aftersalesagent.schemas.contracts import RunLimits

    async def run():
        settings = Settings(
            retail_rag_v2=True, skill_id="after-sales-workflow", skill_version="1.2.0"
        )
        sid = None
        for _ in range(2):
            app = create_app(settings=settings, data_dir=tmp_path, template=retail_template)
            async with app.router.lifespan_context(app):
                hub = app.state.hub
                if sid is None:
                    created = await hub.create("query", RunLimits())
                    sid = created["session"].session_id
                runtime = await hub.get(sid)
                frame, _ = await runtime._snapshot(sid)
                assert frame.skill.version == "1.2.0"
                assert frame.evidence["APP-CAP-V2"]["supplied_by"] == "server_fixed"
                assert hub.retail_index.method == "hybrid"

    asyncio.run(run())


def test_v2_real_mcp_hybrid_parity(store):
    async def run():
        index = RetailV2Index()
        service = RetailService(store)
        view = service.start(RunContext(environment_id=service.environment_id))
        native = ToolRegistry(service, retail_index=index)
        remote = ToolRegistry(
            service,
            retail_index=index,
            provider=MCPRetailProvider(
                service, retail_rag=True, retail_rag_v2=True, retail_retrieval="hybrid"
            ),
        )
        try:
            call = ToolCall(
                call_id="v2-parity",
                name="search_retail_policy",
                arguments={"query": "已送达 delivered 订单自动换货 exchange"},
            )
            ctx = ToolContext(session_id=view.session_id, environment_id=view.environment_id)
            a, b = await native.invoke(call, ctx), await remote.invoke(call, ctx)
            assert a.status == b.status == "ok"
            assert a.data == b.data
            assert "01_service_scope" in {e["chunk_id"] for e in a.data["evidence"]}
        finally:
            await remote.aclose()
            index.close()

    asyncio.run(run())


def test_fixed_capability_citation_and_tamper_rejection(store):
    async def run():
        index = RetailV2Index(method="bm25")
        service = RetailService(store)
        runtime = LangGraphRuntime(
            MockBackend([ModelResponse(content="当前应用没有自动换货工具。[cite:APP-CAP-V2]")]),
            service,
            policy_text="core",
            registry=ToolRegistry(service, retail_index=index),
        )
        try:
            view = await runtime.start(
                RunContext(
                    environment_id=service.environment_id,
                    knowledge_version="retail_sim_v1",
                    knowledge_fingerprint=index.fingerprint,
                )
            )
            result = await runtime.advance(
                view.session_id, MessageEvent(event_id="q", content="你能自动换货吗？")
            )
            assert result.sources[0]["supplied_by"] == "server_fixed"
            assert result.termination_reason is None
            runtime._frames[view.session_id].evidence["APP-CAP-V2"]["text"] = "伪造能力"
            with pytest.raises(ValueError, match="fixed capability mismatch"):
                await runtime.get_state(view.session_id)
        finally:
            index.close()

    asyncio.run(run())


def test_missing_citation_gets_one_retrieval_repair(store):
    async def run():
        index = RetailV2Index(method="bm25")
        service = RetailService(store)
        runtime = LangGraphRuntime(
            MockBackend(
                [
                    ModelResponse(content="已取消订单不能重复取消。"),
                    ModelResponse(content="只允许取消 pending 订单。[cite:POL-07]"),
                ]
            ),
            service,
            policy_text="core",
            registry=ToolRegistry(service, retail_index=index),
        )
        try:
            view = await runtime.start(
                RunContext(
                    environment_id=service.environment_id,
                    knowledge_version="retail_sim_v1",
                    knowledge_fingerprint=index.fingerprint,
                )
            )
            result = await runtime.advance(
                view.session_id, MessageEvent(event_id="q", content="已取消订单还能再取消吗？")
            )
            assert result.termination_reason is None
            assert result.sources[0]["chunk_id"] == "POL-07"
            assert sum(e.payload.get("code") == "policy_repair" for e in result.events) == 1
            assert service.view(view.session_id).pending_action is None
        finally:
            index.close()

    asyncio.run(run())


@pytest.mark.parametrize("method", ["bm25", "hybrid"])
def test_return_dependencies_and_budget(method):
    index = RetailV2Index(method=method)
    try:
        result = index.search(
            RetailSearchArgs(query="已处理 processed 订单还没收到，可以退货 return 吗？"),
            session_id="v2-check",
            call_id="return",
        )
        index.validate_result(
            result,
            session_id="v2-check",
            call_id="return",
            arguments={"query": "已处理 processed 订单还没收到，可以退货 return 吗？"},
        )
        assert {"POL-11", "09_application_protocol"} <= {e["chunk_id"] for e in result["evidence"]}
        assert not result["diagnostics"]["missing_required_ids"]
        index.char_budget = 1
        short = index.search(RetailSearchArgs(query="退货"), session_id="v2", call_id="short")
        assert short["coverage_status"] == "truncated"
        assert "POL-11" in short["diagnostics"]["missing_required_ids"]
    finally:
        index.close()
