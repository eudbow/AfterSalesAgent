"""E2真实子进程/协议测试，无模型API；确认与订单规则仍由原服务负责。"""

import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from mcp import Client

from aftersalesagent.api.app import create_app
from aftersalesagent.business.service import RetailService
from aftersalesagent.config import Settings
from aftersalesagent.evaluation.e1_suite import load
from aftersalesagent.knowledge.corpus import load_corpus
from aftersalesagent.knowledge.retrieval import SQLiteKnowledgeIndex
from aftersalesagent.mcp_transport.adapters import MCPKnowledgeProvider, MCPRetailProvider
from aftersalesagent.mcp_transport.client import MCPConnection
from aftersalesagent.models.scenarios import SCENARIOS
from aftersalesagent.schemas.contracts import RunContext, ToolCall, ToolContext
from aftersalesagent.tools.registry import ToolRegistry

ROOT = Path(__file__).resolve().parents[2]


def error_code(result):
    assert result.error is not None
    return result.error.code


def context(service, sid, key="operation-1"):
    view = service.view(sid)
    return ToolContext(
        session_id=sid,
        environment_id=service.environment_id,
        authenticated_user_id=view.authenticated_user_id,
        idempotency_key=key,
    )


@pytest.mark.parametrize("scenario", ["query", "cancel", "address", "return"])
def test_mcp_http_scenarios_and_confirmation(tmp_path, retail_template, scenario):
    settings = Settings(tool_transport="mcp", knowledge_transport="mcp")
    app = create_app(data_dir=tmp_path / "app", template=retail_template, settings=settings)
    with TestClient(app) as client:
        data = client.post("/sessions", json={"scenario": scenario}).json()
        sid = data["session"]["session_id"]
        headers = {"Authorization": "Bearer " + data["access_token"]}
        assert client.get("/health").json()["tool_transport"] == "mcp"
        first = client.post(
            f"/sessions/{sid}/messages",
            headers=headers,
            json={"event_id": "message-1", "content": data["example_message"]},
        )
        assert first.status_code == 200
        result = first.json()
        if scenario == "query":
            assert result["status"] == "completed"
        else:
            assert result["status"] == "waiting_confirmation", result
            action = result["session"]["pending_action"]
            before = client.get(f"/sessions/{sid}/orders", headers=headers).json()
            current = next(o for o in before["orders"] if o["order_id"] == action["order_id"])
            assert current["status"] == ("delivered" if scenario == "return" else "pending")
            url = f"/sessions/{sid}/actions/{action['action_id']}/confirm"
            body = {"event_id": "confirm-1", "approved": True}
            confirmed = client.post(url, headers=headers, json=body)
            assert confirmed.status_code == 200 and confirmed.json()["status"] == "completed"
            assert client.post(url, headers=headers, json=body).json() == confirmed.json()
        trace = client.get(f"/sessions/{sid}/trace", headers=headers)
        assert trace.status_code == 200
    # 真正关闭旧Hub/连接，再从同数据目录恢复；不复用Python Runtime对象。
    with TestClient(
        create_app(data_dir=tmp_path / "app", template=retail_template, settings=settings)
    ) as reopened:
        assert reopened.get(f"/sessions/{sid}", headers=headers).status_code == 200


def test_mcp_schema_security_and_error_equivalence(store):
    service = RetailService(store)
    sid = service.start(RunContext(environment_id=service.environment_id)).session_id

    async def execute():
        provider = MCPRetailProvider(service)
        registry = ToolRegistry(service, provider=provider)
        connection = provider.connection(sid)
        try:
            await connection.connect()
            assert {s.name for s in service.specs()} <= {t.name for t in connection.discovery}
            spec = next(s for s in service.specs() if s.name == "get_order_details")
            remote = next(t for t in connection.discovery if t.name == spec.name)
            assert remote.input_schema == spec.arguments_schema
            call = ToolCall(call_id="read", name=spec.name, arguments={"order_id": "#W5056519"})
            native = await ToolRegistry(service).invoke(call, context(service, sid))
            actual = await registry.invoke(call, context(service, sid))
            assert actual == native and error_code(actual) == "policy_denied"

            # 标准SDK独立调用，缺少受信能力凭据时不能访问订单。
            async with Client(connection.parameters()) as raw:
                denied = await raw.call_tool(
                    call.name,
                    call.arguments,
                    meta={"aftersales/context": context(service, sid).model_dump(mode="json")},
                )
                assert denied.is_error
                assert denied.structured_content["error"]["code"] == "policy_denied"

            fake = context(service, sid).model_copy(update={"environment_id": "other"})
            assert error_code(await registry.invoke(call, fake)) == "policy_denied"
            name, args = SCENARIOS["query"]["steps"][0]
            assert (
                await registry.invoke(
                    ToolCall(call_id="locate", name=name, arguments=args),
                    context(service, sid, "lookup"),
                )
            ).status == "ok"
            fake = context(service, sid).model_copy(update={"authenticated_user_id": "other"})
            assert error_code(await registry.invoke(call, fake)) == "policy_denied"
            injected = ToolCall(
                call_id="injected",
                name=call.name,
                arguments={**call.arguments, "session_id": sid, "approved": True},
            )
            assert error_code(await registry.invoke(injected, context(service, sid))) == (
                "invalid_arguments"
            )
            unknown = await connection.call("confirm", {}, context=context(service, sid))
            assert unknown["status"] == "error"
            env = connection.parameters().env or {}
            assert "DEEPSEEK_API_KEY" not in env
        finally:
            await registry.aclose()

    asyncio.run(execute())


def test_committed_proposal_lost_response_restart_same_key(store, tmp_path):
    service = RetailService(store)
    sid = service.start(RunContext(environment_id=service.environment_id)).session_id

    async def execute():
        native = ToolRegistry(service)
        name, args = SCENARIOS["cancel"]["steps"][0]
        await native.invoke(
            ToolCall(call_id="lookup", name=name, arguments=args), context(service, sid, "lookup")
        )
        marker = tmp_path / "fault.once"
        provider = MCPRetailProvider(service, fault_after_commit=marker)
        registry = ToolRegistry(service, provider=provider)
        name, args = SCENARIOS["cancel"]["steps"][-1]
        call = ToolCall(call_id="proposal", name=name, arguments=args)
        try:
            with pytest.raises(ConnectionError):
                await registry.invoke(call, context(service, sid, "proposal-key"))
            assert marker.exists()
            pending = service.view(sid).pending_action
            assert pending is not None
            with store.transaction() as connection:
                order = store.order(connection, "#W5056519")
                assert order is not None and order["status"] == "pending"
        finally:
            await registry.aclose()
        # 新SDK客户端、新Server进程；使用原业务键恢复原action。
        provider = MCPRetailProvider(service)
        registry = ToolRegistry(service, provider=provider)
        try:
            replay = await registry.invoke(call, context(service, sid, "proposal-key"))
            assert isinstance(replay.data, dict) and isinstance(replay.data["pending_action"], dict)
            assert replay.status == "ok" and replay.data["pending_action"]["action_id"] == (
                pending.action_id
            )
            assert await registry.invoke(call, context(service, sid, "proposal-key")) == replay
            changed = call.model_copy(update={"arguments": {**args, "reason": "no longer needed"}})
            assert (
                error_code(await registry.invoke(changed, context(service, sid, "proposal-key")))
                == "invalid_arguments"
            )
            # 宿主的原确认入口提交，重放不产生第二次退款。
            first = service.confirm(sid, pending.action_id, True)
            assert first.status == "ok" and service.confirm(sid, pending.action_id, True) == first
            with store.transaction() as connection:
                order = store.order(connection, "#W5056519")
                assert order is not None
                assert order["status"] == "cancelled" and len(order["payment_history"]) == 2
        finally:
            await registry.aclose()

    asyncio.run(execute())


def test_all_e1_evidence_native_mcp_equivalent():
    chunks, _ = load_corpus(ROOT)

    async def execute():
        native = SQLiteKnowledgeIndex(chunks)
        remote = MCPKnowledgeProvider(SQLiteKnowledgeIndex(chunks))
        try:
            for split in ("dev", "validation", "test"):
                cases, _ = load(ROOT, split)
                for case in cases:
                    p = case["public"]
                    for full in (False, True):
                        expected = await native.provide(p["question"], p["filters"], 5, full=full)
                        actual = await remote.provide(p["question"], p["filters"], 5, full=full)
                        assert actual == expected, (p["task_id"], full)
            assert remote.remote.protocol_version
        finally:
            await remote.aclose()
            remote.close()
            native.close()

    asyncio.run(execute())


@pytest.mark.parametrize("mode", ["rag", "full"])
def test_mcp_knowledge_http_preserves_sources_and_cache(tmp_path, retail_template, mode):
    app = create_app(
        data_dir=tmp_path,
        template=retail_template,
        settings=Settings(knowledge_transport="mcp"),
    )
    with TestClient(app) as client:
        assert client.get("/knowledge/status").json()["transport"] == "mcp"
        body = {"request_id": "knowledge-1", "question": "银行卡退货凭证", "mode": mode}
        first = client.post("/knowledge/ask", json=body)
        assert first.status_code == 200
        data = first.json()
        assert data["sources"] and data["sources"][0]["effective_from"] is None
        assert data["backend"] == "mock_demo"
        assert client.post("/knowledge/ask", json=body).json() == data
        assert "model_request" not in data


def test_harness_retries_original_key_after_mcp_process_exit(store, tmp_path):
    from aftersalesagent.harness.execution import invoke_safely

    service = RetailService(store)
    sid = service.start(RunContext(environment_id=service.environment_id)).session_id

    async def execute():
        name, args = SCENARIOS["cancel"]["steps"][0]
        await ToolRegistry(service).invoke(
            ToolCall(call_id="lookup", name=name, arguments=args), context(service, sid, "lookup")
        )
        provider = MCPRetailProvider(service, fault_after_commit=tmp_path / "fault.once")
        registry = ToolRegistry(service, provider=provider)
        name, args = SCENARIOS["cancel"]["steps"][-1]
        call = ToolCall(call_id="proposal", name=name, arguments=args)
        attempts = []

        async def record(code):
            attempts.append(code)

        try:
            result = await invoke_safely(
                lambda: registry.invoke(call, context(service, sid, "same-key")),
                call.call_id,
                on_retry=record,
            )
            assert result.status == "ok", result
            assert attempts == ["infrastructure_error"]
            assert service.view(sid).pending_action is not None
        finally:
            await registry.aclose()

    asyncio.run(execute())


def test_cancellation_waits_for_mcp_result_before_releasing_caller(store, tmp_path):
    import time

    service = RetailService(store)
    sid = service.start(RunContext(environment_id=service.environment_id)).session_id

    async def execute():
        name, args = SCENARIOS["cancel"]["steps"][0]
        await ToolRegistry(service).invoke(
            ToolCall(call_id="lookup", name=name, arguments=args), context(service, sid, "lookup")
        )
        marker = tmp_path / "delay.once"
        connection = MCPConnection(
            database=store.path,
            session_id=sid,
            allow_proposals=True,
            fault_after_commit=marker,
            fault_delay_seconds=1,
        )
        try:
            await connection.connect()
            name, args = SCENARIOS["cancel"]["steps"][-1]
            task = asyncio.create_task(
                connection.call(
                    name, args, context=context(service, sid, "same-key"), call_id="proposal"
                )
            )

            async def wait_for_commit():
                while not marker.exists():
                    await asyncio.sleep(0.02)

            await asyncio.wait_for(wait_for_commit(), 5)
            started = time.monotonic()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert time.monotonic() - started >= 0.5
            assert service.view(sid).pending_action is not None
        finally:
            await connection.aclose()

    asyncio.run(execute())


def test_superseded_mcp_proposal_stays_stale_and_rejection_does_not_write(store):
    service = RetailService(store)
    sid = service.start(RunContext(environment_id=service.environment_id)).session_id

    async def execute():
        provider = MCPRetailProvider(service)
        registry = ToolRegistry(service, provider=provider)
        try:
            name, args = SCENARIOS["cancel"]["steps"][0]
            await registry.invoke(
                ToolCall(call_id="lookup", name=name, arguments=args),
                context(service, sid, "lookup"),
            )
            name, args = SCENARIOS["cancel"]["steps"][-1]
            call = ToolCall(call_id="proposal", name=name, arguments=args)
            await registry.invoke(call, context(service, sid, "old-key"))
            old = service.view(sid).pending_action
            assert old is not None
            await registry.invoke(call, context(service, sid, "new-key"))
            current = service.view(sid).pending_action
            assert current is not None and current.action_id != old.action_id
            assert error_code(service.confirm(sid, old.action_id, True)) == "confirmation_stale"
            assert service.confirm(sid, current.action_id, False).status == "ok"
            with store.transaction() as connection:
                order = store.order(connection, "#W5056519")
                assert order is not None and order["status"] == "pending"
        finally:
            await registry.aclose()

    asyncio.run(execute())
