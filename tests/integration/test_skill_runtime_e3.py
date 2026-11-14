"""真实 Runtime/SQLite/MCP 与 Mock 模型：只验工程机制，不估计模型收益。"""

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import SecretStr
from sqlalchemy import func, select

from aftersalesagent.api.app import create_app
from aftersalesagent.business.database import sessions
from aftersalesagent.business.service import RetailService
from aftersalesagent.config import Settings
from aftersalesagent.harness.journal import Journal
from aftersalesagent.models.deepseek import DeepSeekBackend, DeepSeekOptions
from aftersalesagent.models.mock import MockBackend
from aftersalesagent.policies.application import load_application_policy
from aftersalesagent.runtime.langgraph_runtime import LangGraphRuntime
from aftersalesagent.schemas.contracts import MessageEvent, ModelResponse, RunContext
from aftersalesagent.skills.base import NullSkillProvider, text_hash
from aftersalesagent.skills.registry import SkillRegistry

ROOT = Path(__file__).resolve().parents[2]
SKILL_ID = "after-sales-workflow"


def make_registry(path):
    reg = SkillRegistry(path)
    reg.install(ROOT / "skills", SKILL_ID, "1.0.0")
    return reg


def make_test_version(reg, tmp_path):
    original = asyncio.run(reg.load(SKILL_ID, "1.0.0"))
    data = original.metadata().model_dump(mode="json")
    text = original.text + "\n测试专用 v2，不是生产发布。\n"
    data.update(
        version="1.0.1",
        parent_version="1.0.0",
        content_sha256=text_hash(text),
        change_notes="测试快照和默认指针。",
    )
    folder = tmp_path / "test-bundles" / SKILL_ID / "1.0.1"
    folder.mkdir(parents=True)
    (folder / "manifest.json").write_text(json.dumps(data), encoding="utf-8")
    (folder / "SKILL.md").write_bytes(text.encode("utf-8"))
    reg.install(folder.parents[1], SKILL_ID, "1.0.1")


def test_no_skill_keeps_original_prompt_and_never_loads_provider(store, tmp_path):
    class ForbiddenProvider(NullSkillProvider):
        async def load(self, skill_id: str, version: str):
            raise AssertionError("No-Skill must not invoke provider")

    async def scenario():
        backend = MockBackend([ModelResponse(content="ok")])
        runtime = LangGraphRuntime(
            backend,
            RetailService(store),
            policy_text="original core",
            skill_provider=ForbiddenProvider(),
        )
        sid = (await runtime.start(RunContext(environment_id=store.path.stem))).session_id
        await runtime.advance(sid, MessageEvent(event_id="m", content="hello"))
        assert [m.content for m in backend.requests[0].messages] == ["original core", "hello"]
        frame, _ = await runtime._snapshot(sid)
        assert frame.skill is None and frame.context.skill_version == "none"

    asyncio.run(scenario())


@pytest.mark.parametrize("mismatch", ["policy", "tools", "missing_provider", "missing_version"])
def test_incompatible_skill_fails_before_creating_business_session(store, tmp_path, mismatch):
    async def scenario():
        reg = make_registry(tmp_path / "skills.sqlite")
        service = RetailService(store)
        policy = load_application_policy(ROOT)
        runtime = LangGraphRuntime(
            MockBackend([]),
            service,
            policy_text=policy + ("changed" if mismatch == "policy" else ""),
            skill_provider=None if mismatch == "missing_provider" else reg,
        )
        if mismatch == "tools":
            runtime.registry.specs = lambda: service.specs()[:-1]
        with pytest.raises(ValueError):
            await runtime.start(
                RunContext(
                    environment_id=store.path.stem,
                    skill_id=SKILL_ID,
                    skill_version="9.0.0" if mismatch == "missing_version" else "1.0.0",
                )
            )
        with store.transaction() as db:
            assert db.scalar(select(func.count()).select_from(sessions)) == 0

    asyncio.run(scenario())


def test_snapshot_pinning_restart_and_legacy_checkpoint(store, tmp_path):
    reg = make_registry(tmp_path / "skills.sqlite")
    make_test_version(reg, tmp_path)
    reg.activate(SKILL_ID, "1.0.0", expected_current="none", reason="test-start")
    checkpoint = str(tmp_path / "checkpoints.sqlite")
    journal = Journal(tmp_path / "journal.sqlite")

    async def scenario():
        policy = load_application_policy(ROOT)
        service = RetailService(store)
        async with AsyncSqliteSaver.from_conn_string(checkpoint) as saver:
            runtime = LangGraphRuntime(
                MockBackend([]),
                service,
                policy_text=policy,
                checkpointer=saver,
                journal=journal,
                skill_provider=reg,
            )
            old = await runtime.start(
                RunContext(
                    environment_id=store.path.stem, skill_id=SKILL_ID, skill_version="released"
                )
            )
            frame, _ = await runtime._snapshot(old.session_id)
            assert frame.context.skill_version == "1.0.0"
            assert frame.skill is not None
            with store.transaction() as db:
                stored = db.scalar(
                    select(sessions.c.context_json).where(sessions.c.session_id == old.session_id)
                )
            assert stored["skill_sha256"] == frame.skill.content_sha256
            reg.activate(SKILL_ID, "1.0.1", expected_current="1.0.0", reason="test-switch")
            new = await runtime.start(
                RunContext(
                    environment_id=store.path.stem, skill_id=SKILL_ID, skill_version="released"
                )
            )
            newer, _ = await runtime._snapshot(new.session_id)
            assert newer.context.skill_version == "1.0.1"
            legacy = await runtime.start(RunContext(environment_id=store.path.stem))
            snapshot = await runtime.graph.aget_state(runtime._config(legacy.session_id))
            raw = snapshot.values["frame"]
            raw.pop("skill")
            for field in ("skill_id", "skill_sha256", "skill_snapshot_sha256"):
                raw["context"].pop(field)
            await runtime.graph.aupdate_state(
                runtime._config(legacy.session_id), {"frame": raw}, as_node="execute"
            )
        # 关闭再重开文件 Checkpointer，移除版本库后仍可从会话快照继续。
        reg.path.unlink()
        backend = MockBackend([ModelResponse(content="continued"), ModelResponse(content="legacy")])
        async with AsyncSqliteSaver.from_conn_string(checkpoint) as saver:
            restored = LangGraphRuntime(
                backend, service, policy_text=policy, checkpointer=saver, journal=journal
            )
            await restored.advance(old.session_id, MessageEvent(event_id="m", content="continue"))
            request = backend.requests[0]
            assert request.messages[1].content == frame.skill.prompt()
            assert "测试专用 v2" not in request.messages[1].content
            await restored.advance(
                legacy.session_id, MessageEvent(event_id="m", content="old format")
            )
            assert len(backend.requests[1].messages) == 2
            events = journal.events(old.session_id)
            first = next(e for e in events if e.kind == "model_request")
            assert first.payload["skill_sha256"] == frame.skill.content_sha256
            assert first.payload["skill_snapshot_sha256"] == frame.context.skill_snapshot_sha256
            broken = await restored.graph.aget_state(restored._config(old.session_id))
            raw = broken.values["frame"]
            raw["messages"][1]["content"] += "tampered"
            await restored.graph.aupdate_state(
                restored._config(old.session_id), {"frame": raw}, as_node="execute"
            )
            with pytest.raises(ValueError, match="snapshot or prompt"):
                await restored.get_state(old.session_id)
        journal.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("transport", ["native", "mcp"])
def test_skill_api_confirmation_and_restart_to_no_skill(tmp_path, retail_template, transport):
    directory = tmp_path / "app"
    settings = Settings(tool_transport=transport, skill_id=SKILL_ID, skill_version="1.0.0")
    with TestClient(
        create_app(data_dir=directory, template=retail_template, settings=settings)
    ) as client:
        body = client.post("/sessions", json={"scenario": "cancel"}).json()
        sid = body["session"]["session_id"]
        headers = {"Authorization": "Bearer " + body["access_token"]}
        result = client.post(
            f"/sessions/{sid}/messages",
            headers=headers,
            json={"event_id": "m", "content": body["example_message"]},
        ).json()
        assert result["status"] == "waiting_confirmation"
        action = result["session"]["pending_action"]
        order = next(
            o
            for o in client.get(f"/sessions/{sid}/orders", headers=headers).json()["orders"]
            if o["order_id"] == action["order_id"]
        )
        assert order["status"] == "pending"
        events = client.get(f"/sessions/{sid}/trace", headers=headers).json()["events"]
        request = next(e["payload"] for e in events if e["kind"] == "model_request")
        assert request["skill_version"] == "1.0.0"
        assert len(request["skill_sha256"]) == 64
    # 默认配置改为无 Skill 不应阻止旧 Skill 会话恢复，也不能改掉它。
    with TestClient(
        create_app(
            data_dir=directory,
            template=retail_template,
            settings=Settings(tool_transport=transport),
        )
    ) as reopened:
        url = f"/sessions/{sid}/actions/{action['action_id']}/confirm"
        body = {"event_id": "confirm", "approved": True}
        result = reopened.post(url, headers=headers, json=body)
        assert result.status_code == 200 and result.json()["status"] == "completed"
        assert reopened.post(url, headers=headers, json=body).json() == result.json()
        trace = reopened.get(f"/sessions/{sid}/trace", headers=headers).json()["events"]
        assert sum(e["kind"] == "state_change" for e in trace) == 1
        end = [e["payload"] for e in trace if e["kind"] == "turn_end"][-1]
        assert end["skill_version"] == "1.0.0"
        assert end["skill_sha256"] == request["skill_sha256"]
        # 新会话默认仍是原始无 Skill。
        assert reopened.get("/health").json()["skill_selector"] == "none"


def test_deepseek_adapter_receives_pinned_skill_without_real_network(store, tmp_path):
    seen = []

    def respond(request):
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "offline-e3",
                "model": "test",
                "choices": [
                    {"finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}
                ],
                "usage": {"prompt_tokens": 321, "completion_tokens": 2},
            },
        )

    async def scenario():
        backend = DeepSeekBackend(
            DeepSeekOptions(model="test"),
            SecretStr("offline-only-e3"),
            transport=httpx.MockTransport(respond),
        )
        runtime = LangGraphRuntime(
            backend,
            RetailService(store),
            policy_text=load_application_policy(ROOT),
            skill_provider=make_registry(tmp_path / "skills.sqlite"),
        )
        sid = (
            await runtime.start(
                RunContext(
                    environment_id=store.path.stem,
                    backend="deepseek",
                    model_name="test",
                    skill_id=SKILL_ID,
                    skill_version="1.0.0",
                )
            )
        ).session_id
        result = await runtime.advance(sid, MessageEvent(event_id="m", content="hello"))
        assert result.status == "completed"
        assert seen[0]["messages"][0]["content"] == load_application_policy(ROOT)
        assert seen[0]["messages"][1]["content"].startswith("Supplemental workflow guidance")
        assert len(seen[0]["tools"]) == len(runtime.registry.specs())
        assert runtime.journal.usage(sid)["reported_tokens"] == 323
        await backend.aclose()

    asyncio.run(scenario())


def test_running_hub_resolves_release_per_new_session_and_pins_old_sessions(
    tmp_path, retail_template
):
    directory = tmp_path / "app"
    reg = make_registry(directory / "skills/registry.sqlite")
    make_test_version(reg, tmp_path)
    reg.activate(SKILL_ID, "1.0.0", expected_current="none", reason="manual baseline")
    settings = Settings(skill_id=SKILL_ID, skill_version="released")
    with TestClient(
        create_app(data_dir=directory, template=retail_template, settings=settings)
    ) as client:
        first = client.post("/sessions", json={"scenario": "query"}).json()
        reg.activate(SKILL_ID, "1.0.1", expected_current="1.0.0", reason="test-only switch")
        second = client.post("/sessions", json={"scenario": "query"}).json()
        reg.activate(SKILL_ID, "1.0.0", expected_current="1.0.1", reason="test-only rollback")
        third = client.post("/sessions", json={"scenario": "query"}).json()
        for session, version in zip(
            (first, second, third), ("1.0.0", "1.0.1", "1.0.0"), strict=True
        ):
            sid = session["session"]["session_id"]
            headers = {"Authorization": "Bearer " + session["access_token"]}
            # 三个会话都在回退完成后才第一次调用模型。
            response = client.post(
                f"/sessions/{sid}/messages",
                headers=headers,
                json={"event_id": "m", "content": session["example_message"]},
            )
            assert response.status_code == 200 and response.json()["status"] == "completed"
            events = client.get(f"/sessions/{sid}/trace", headers=headers).json()["events"]
            requests = [e["payload"] for e in events if e["kind"] == "model_request"]
            assert requests and all(p["skill_version"] == version for p in requests)
    assert [d["version"] for d in reg.history()] == ["1.0.0", "1.0.1", "1.0.0"]
