"""一个进程管理少量隔离会话；路径和业务身份只能由服务端建立。"""

from __future__ import annotations

import asyncio
import secrets
from collections.abc import Callable
from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver
from sqlalchemy import select

from aftersalesagent.business.database import RetailStore, clone_database, metadata, sessions
from aftersalesagent.business.service import RetailService
from aftersalesagent.harness.execution import run_sync
from aftersalesagent.harness.journal import Journal
from aftersalesagent.mcp_transport.adapters import MCPRetailProvider
from aftersalesagent.models.base import ModelBackend
from aftersalesagent.models.scenarios import SCENARIOS, ScenarioMockBackend
from aftersalesagent.runtime.langgraph_runtime import LangGraphRuntime
from aftersalesagent.schemas.contracts import RunContext, RunLimits
from aftersalesagent.skills.base import SkillProvider
from aftersalesagent.tools.registry import ToolRegistry


class SessionHub:
    def __init__(
        self,
        template: Path,
        data_dir: Path,
        policy: str,
        journal: Journal,
        saver: BaseCheckpointSaver,
        backend_factory: Callable[[str], ModelBackend] | None = None,
        *,
        backend_name: str = "mock",
        model_name: str | None = None,
        temperature: float = 0,
        max_output_tokens: int = 1024,
        tool_transport: str = "native",
        skill_provider: SkillProvider | None = None,
        skill_id: str | None = None,
        skill_version: str = "none",
    ):
        self.template, self.data_dir, self.policy = template, data_dir, policy
        self.journal, self.saver = journal, saver
        self.backend_factory = backend_factory or ScenarioMockBackend
        self.backend_name, self.model_name = backend_name, model_name
        self.temperature, self.max_output_tokens = temperature, max_output_tokens
        self.tool_transport = tool_transport
        self.skill_provider, self.skill_id, self.skill_version = (
            skill_provider,
            skill_id,
            skill_version,
        )
        self._runtimes: dict[str, LangGraphRuntime] = {}
        self._lock = asyncio.Lock()

    def _runtime(self, store: RetailStore, scenario: str) -> LangGraphRuntime:
        service = RetailService(store)
        provider = (
            MCPRetailProvider(
                service, trace_path=self.data_dir / "traces" / (store.path.stem + "_mcp.jsonl")
            )
            if self.tool_transport == "mcp"
            else None
        )
        return LangGraphRuntime(
            self.backend_factory(scenario),
            service,
            registry=ToolRegistry(service, provider=provider),
            skill_provider=self.skill_provider,
            policy_text=self.policy,
            checkpointer=self.saver,
            journal=self.journal,
            trace_directory=self.data_dir / "traces",
        )

    async def create(self, scenario: str, limits: RunLimits) -> dict:
        import uuid

        environment = "episode_" + uuid.uuid4().hex
        store = await run_sync(
            lambda: clone_database(
                self.template, self.data_dir / "databases" / (environment + ".sqlite")
            )
        )
        # 只增建应用表，兼容已有M1模板，不改上游和既有订单字段。
        await run_sync(lambda: metadata.create_all(store.engine))
        runtime = self._runtime(store, scenario)
        try:
            session = await runtime.start(
                RunContext.model_validate(
                    {
                        "environment_id": environment,
                        "limits": limits,
                        "backend": self.backend_name,
                        "tool_transport": self.tool_transport,
                        "skill_id": self.skill_id,
                        "skill_version": self.skill_version,
                        "model_name": self.model_name,
                        "temperature": self.temperature,
                        "max_output_tokens": self.max_output_tokens,
                    }
                )
            )
            token = secrets.token_urlsafe(32)
            await run_sync(
                lambda: self.journal.add_session(session.session_id, environment, scenario, token)
            )
        except BaseException:
            store.close()
            raise
        self._runtimes[session.session_id] = runtime
        return {
            "session": session,
            "access_token": token,
            "scenario": scenario,
            "example_message": SCENARIOS[scenario]["prompt"],
        }

    async def get(self, sid: str) -> LangGraphRuntime:
        async with self._lock:
            if sid in self._runtimes:
                return self._runtimes[sid]
            entry = await run_sync(lambda: self.journal.session(sid))
            if entry is None:
                raise ValueError("Unknown session")
            # environment来自持久化服务端目录，不来自HTTP路径参数。
            path = self.data_dir / "databases" / (entry["environment"] + ".sqlite")
            if not path.is_file():
                raise ValueError("Session environment is missing")
            store = RetailStore(path)

            def stored_context():
                with store.transaction() as connection:
                    payload = connection.scalar(
                        select(sessions.c.context_json).where(sessions.c.session_id == sid)
                    )
                    return RunContext.model_validate(payload)

            context = await run_sync(stored_context)
            if (
                context.tool_transport != self.tool_transport
                or context.backend != self.backend_name
                or context.model_name != self.model_name
                or context.temperature != self.temperature
                or context.max_output_tokens != self.max_output_tokens
            ):
                store.close()
                raise ValueError("Resume with the session's original backend/model configuration")
            runtime = self._runtime(store, entry["scenario"])
            try:
                await runtime.get_state(sid)
            except BaseException:
                store.close()
                raise
            self._runtimes[sid] = runtime
            return runtime

    async def close(self) -> None:
        for runtime in self._runtimes.values():
            await runtime.registry.aclose()
            closer = getattr(runtime.backend, "aclose", None)
            if closer:
                await closer()
            await run_sync(runtime.service.store.close)
        self._runtimes.clear()
