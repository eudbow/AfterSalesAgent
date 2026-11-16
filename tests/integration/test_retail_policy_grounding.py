"""Citation provenance and backend proposal evidence, without paid calls."""

import asyncio

import pytest

from aftersalesagent.business.service import RetailService
from aftersalesagent.harness.journal import EventConflict
from aftersalesagent.knowledge.retail import RetailPolicyIndex
from aftersalesagent.models.mock import MockBackend
from aftersalesagent.runtime.langgraph_runtime import LangGraphRuntime
from aftersalesagent.schemas.contracts import (
    ConfirmationEvent,
    MessageEvent,
    ModelResponse,
    RunContext,
    ToolCall,
)
from aftersalesagent.tools.registry import ToolRegistry


def test_unseen_citation_is_hidden_but_raw_response_retained(store):
    class Backend:
        async def generate(self, request):
            return ModelResponse(content="保证今天到账。[cite:POL-07]")

    async def run():
        service = RetailService(store)
        index = RetailPolicyIndex()
        runtime = LangGraphRuntime(
            Backend(),
            service,
            policy_text="policy",
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
                view.session_id, MessageEvent(event_id="q", content="多久到账")
            )
            assert result.termination_reason == "invalid_policy_citation"
            assert "保证今天" not in result.reply and result.sources == []
            frame, _ = await runtime._snapshot(view.session_id)
            assert "保证今天" in frame.response.content
        finally:
            index.close()

    asyncio.run(run())


def test_missing_policy_citation_is_hidden_but_plain_order_status_is_allowed(store):
    async def run():
        service = RetailService(store)
        index = RetailPolicyIndex()
        backend = MockBackend(
            [
                ModelResponse(content="已取消订单不能再取消。"),
                ModelResponse(content="订单状态为 cancelled。"),
            ]
        )
        runtime = LangGraphRuntime(
            backend, service, policy_text="core", registry=ToolRegistry(service, retail_index=index)
        )
        try:
            first = await runtime.start(
                RunContext(
                    environment_id=service.environment_id,
                    knowledge_version="retail_sim_v1",
                    knowledge_fingerprint=index.fingerprint,
                )
            )
            result = await runtime.advance(
                first.session_id,
                MessageEvent(event_id="policy", content="已取消订单还能再取消一次吗？"),
            )
            assert result.termination_reason == "missing_policy_citation"
            assert "不能再取消" not in result.reply
            frame, _ = await runtime._snapshot(first.session_id)
            assert "不能再取消" in frame.response.content

            second = await runtime.start(
                RunContext(
                    environment_id=service.environment_id,
                    knowledge_version="retail_sim_v1",
                    knowledge_fingerprint=index.fingerprint,
                )
            )
            status = await runtime.advance(
                second.session_id,
                MessageEvent(event_id="status", content="请查一下订单当前状态。"),
            )
            assert status.reply == "订单状态为 cancelled。"
            assert status.termination_reason is None
        finally:
            index.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "fault,expected_code", [(ValueError, "infrastructure_error"), (TimeoutError, "timeout")]
)
def test_retrieval_fault_does_not_block_later_order_lookup(
    store, monkeypatch, fault, expected_code
):
    async def run():
        service = RetailService(store)
        index = RetailPolicyIndex()
        responses = [
                ModelResponse(
                    tool_calls=[
                        ToolCall(
                            call_id="bad-search",
                            name="search_retail_policy",
                            arguments={"query": "退货时效"},
                        )
                    ]
                ),
                *(
                    [ModelResponse(content="资料暂时无法核实。")]
                    if fault is TimeoutError
                    else []
                ),
                ModelResponse(
                    tool_calls=[
                        ToolCall(
                            call_id="auth",
                            name="find_user_id_by_name_zip",
                            arguments={
                                "first_name": "Yara",
                                "last_name": "Muller",
                                "zip": "85041",
                            },
                        )
                    ]
                ),
                ModelResponse(
                    tool_calls=[
                        ToolCall(
                            call_id="order",
                            name="get_order_details",
                            arguments={"order_id": "#W5056519"},
                        )
                    ]
                ),
                ModelResponse(content="订单 #W5056519 当前状态为 pending。"),
            ]
        backend = MockBackend(responses)
        runtime = LangGraphRuntime(
            backend, service, policy_text="core", registry=ToolRegistry(service, retail_index=index)
        )
        try:
            view = await runtime.start(
                RunContext(
                    environment_id=service.environment_id,
                    knowledge_version="retail_sim_v1",
                    knowledge_fingerprint=index.fingerprint,
                )
            )

            def fail(*_args, **_kwargs):
                raise fault("Index unavailable")

            original_search = index.search
            monkeypatch.setattr(index, "search", fail)
            failed = await runtime.advance(
                view.session_id, MessageEvent(event_id="policy", content="退货时效是什么？")
            )
            assert failed.status == "failed"
            assert failed.termination_reason == expected_code
            if fault is TimeoutError:
                assert any(event.kind == "retry" for event in failed.events)
                with pytest.raises(EventConflict, match="original event_id"):
                    await runtime.advance(
                        view.session_id,
                        MessageEvent(event_id="status", content="请查询订单状态"),
                    )
                monkeypatch.setattr(index, "search", original_search)
                recovered = await runtime.advance(
                    view.session_id,
                    MessageEvent(event_id="policy", content="退货时效是什么？"),
                )
                assert recovered.status == "completed"
            lookup = await runtime.advance(
                view.session_id,
                MessageEvent(
                    event_id="status",
                    content="我是 Yara Muller，邮编 85041。查订单 #W5056519 状态。",
                ),
            )
            assert lookup.status == "completed"
            assert "pending" in lookup.reply
            assert lookup.termination_reason is None
        finally:
            index.close()

    asyncio.run(run())


def test_other_session_evidence_cannot_be_cited(store):
    async def run():
        service = RetailService(store)
        index = RetailPolicyIndex()
        backend = MockBackend(
            [
                ModelResponse(
                    tool_calls=[
                        ToolCall(
                            call_id="search-a",
                            name="search_retail_policy",
                            arguments={"query": "退货退款到账时间"},
                        )
                    ]
                ),
                ModelResponse(content="资料没有规定。[cite:06_refund_methods]"),
                ModelResponse(content="资料没有规定。[cite:06_refund_methods]"),
            ]
        )
        runtime = LangGraphRuntime(
            backend, service, policy_text="core", registry=ToolRegistry(service, retail_index=index)
        )
        try:
            context = RunContext(
                environment_id=service.environment_id,
                knowledge_version="retail_sim_v1",
                knowledge_fingerprint=index.fingerprint,
            )
            first = await runtime.start(context)
            answer = await runtime.advance(
                first.session_id,
                MessageEvent(event_id="a", content="退货退款多久到账？"),
            )
            assert any(s["chunk_id"] == "06_refund_methods" for s in answer.sources)
            second = await runtime.start(context)
            leak = await runtime.advance(
                second.session_id,
                MessageEvent(event_id="b", content="退货退款多久到账？"),
            )
            assert leak.termination_reason == "invalid_policy_citation"
            assert leak.sources == []
        finally:
            index.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "field,value", [("version", "old"), ("scope", "ikea_cn"), ("text", "changed")]
)
def test_stale_or_changed_checkpoint_evidence_is_rejected(store, field, value):
    async def run():
        service = RetailService(store)
        index = RetailPolicyIndex()
        backend = MockBackend(
            [
                ModelResponse(
                    tool_calls=[
                        ToolCall(
                            call_id="search",
                            name="search_retail_policy",
                            arguments={"query": "退货退款到账时间"},
                        )
                    ]
                ),
                ModelResponse(content="资料未规定。[cite:06_refund_methods]"),
            ]
        )
        runtime = LangGraphRuntime(
            backend, service, policy_text="core", registry=ToolRegistry(service, retail_index=index)
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
                view.session_id, MessageEvent(event_id="first", content="退货退款多久到账？")
            )
            assert result.sources
            runtime._frames[view.session_id].evidence["06_refund_methods"][field] = value
            with pytest.raises(ValueError, match="checkpoint evidence content mismatch"):
                await runtime.get_state(view.session_id)
        finally:
            index.close()

    asyncio.run(run())


@pytest.mark.parametrize("citation", ["POL-07", "invented-policy"])
def test_pending_card_keeps_checked_policy_explanation_and_separate_confirmation(
    store, citation
):
    async def run():
        service = RetailService(store)
        index = RetailPolicyIndex()
        backend = MockBackend(
            [
                ModelResponse(
                    tool_calls=[
                        ToolCall(
                            call_id="auth",
                            name="find_user_id_by_name_zip",
                            arguments={
                                "first_name": "Yara",
                                "last_name": "Muller",
                                "zip": "85041",
                            },
                        )
                    ]
                ),
                ModelResponse(
                    tool_calls=[
                        ToolCall(
                            call_id="order",
                            name="get_order_details",
                            arguments={"order_id": "#W5056519"},
                        )
                    ]
                ),
                ModelResponse(
                    tool_calls=[
                        ToolCall(
                            call_id="policy",
                            name="search_retail_policy",
                            arguments={"query": "cancel pending order reason ordered by mistake"},
                        )
                    ]
                ),
                ModelResponse(
                    content=(
                        f"待处理订单可以因下错单取消。[cite:{citation}]"
                        "\n\n确认后你会收到一封邮件。"
                    ),
                    tool_calls=[
                        ToolCall(
                            call_id="prepare",
                            name="cancel_pending_order",
                            arguments={
                                "order_id": "#W5056519",
                                "reason": "ordered by mistake",
                            },
                        )
                    ],
                ),
            ]
        )
        runtime = LangGraphRuntime(
            backend, service, policy_text="core", registry=ToolRegistry(service, retail_index=index)
        )
        try:
            view = await runtime.start(
                RunContext(
                    environment_id=service.environment_id,
                    knowledge_version="retail_sim_v1",
                    knowledge_fingerprint=index.fingerprint,
                )
            )
            prepared = await runtime.advance(
                view.session_id,
                MessageEvent(event_id="prepare-event", content="请取消订单，原因是下错单。"),
            )
            assert prepared.status == "waiting_confirmation"
            assert "待确认操作" in prepared.reply
            assert "邮件" not in prepared.reply
            assert any(
                event.kind == "model_error"
                and event.payload.get("code") == "unsupported_app_promise_filtered"
                for event in prepared.events
            )
            if citation == "POL-07":
                assert "待处理订单可以因下错单取消" in prepared.reply
                assert [source["chunk_id"] for source in prepared.sources] == ["POL-07"]
                assert prepared.termination_reason is None
            else:
                assert "待处理订单可以因下错单取消" not in prepared.reply
                assert prepared.sources == []
                assert prepared.termination_reason == "invalid_policy_citation"
            with store.transaction() as connection:
                assert store.order(connection, "#W5056519")["status"] == "pending"
            action = prepared.session.pending_action
            confirmed = await runtime.advance(
                view.session_id,
                ConfirmationEvent(
                    event_id="confirm-event", action_id=action.action_id, approved=True
                ),
            )
            assert confirmed.status == "completed"
            with store.transaction() as connection:
                assert store.order(connection, "#W5056519")["status"] == "cancelled"
        finally:
            index.close()

    asyncio.run(run())


def test_proposal_has_persisted_backend_basis_without_retrieval(store):
    service = RetailService(store)
    index = RetailPolicyIndex()
    service.policy_basis = index.action_basis
    try:
        view = service.start(RunContext(environment_id=service.environment_id))
        sid = view.session_id
        service.dispatch(
            sid,
            ToolCall(
                call_id="auth",
                name="find_user_id_by_name_zip",
                arguments={"first_name": "Yara", "last_name": "Muller", "zip": "85041"},
            ),
        )
        result = service.dispatch(
            sid,
            ToolCall(
                call_id="prepare",
                name="cancel_pending_order",
                arguments={"order_id": "#W5056519", "reason": "ordered by mistake"},
            ),
            idempotency_key="prepare-key",
        )
        assert result.status == "ok"
        action = service.view(sid).pending_action
        assert action.policy_basis[0]["rule_id"] == "POL-07"
        assert action.policy_basis[0]["supplied_by"] == "backend_rule_mapping"
        # A separate service reads the snapshot; it does not recompute policy text.
        restored = RetailService(store).view(sid).pending_action
        assert restored == action
        rejected = service.confirm(sid, action.action_id, False)
        assert rejected.status == "ok"
        order = service.dispatch(
            sid,
            ToolCall(
                call_id="order", name="get_order_details", arguments={"order_id": "#W5056519"}
            ),
        )
        assert order.data["status"] == "pending"
    finally:
        index.close()


def test_multi_read_response_executes_only_first_read(store):
    class Backend:
        async def generate(self, request):
            if request.messages[-1].role != "tool":
                return ModelResponse(
                    tool_calls=[
                        ToolCall(
                            call_id="first",
                            name="search_retail_policy",
                            arguments={"query": "退货到账时间"},
                        ),
                        ToolCall(
                            call_id="second",
                            name="get_order_details",
                            arguments={"order_id": "#W1023987"},
                        ),
                    ]
                )
            return ModelResponse(content="资料未规定。[cite:06_refund_methods]")

    async def run():
        service = RetailService(store)
        index = RetailPolicyIndex()
        runtime = LangGraphRuntime(
            Backend(),
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
                view.session_id, MessageEvent(event_id="multi", content="退货多久到账")
            )
            assert result.status == "completed" and result.sources
            calls = [e.payload.get("name") for e in result.events if e.kind == "tool_call"]
            assert calls == ["search_retail_policy"]
            assert any(
                e.kind == "model_error" and e.payload.get("code") == "extra_read_calls_ignored"
                for e in result.events
            )
        finally:
            index.close()

    asyncio.run(run())
