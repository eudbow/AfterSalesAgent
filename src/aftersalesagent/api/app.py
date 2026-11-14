"""FastAPI本地应用：只暴露应用契约，不暴露LangGraph State和数据库路径。"""

from __future__ import annotations

import hashlib
import secrets
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from aftersalesagent.api.hub import SessionHub
from aftersalesagent.api.knowledge import router as knowledge_router
from aftersalesagent.business.database import import_pinned_database
from aftersalesagent.config import Settings
from aftersalesagent.harness.execution import run_sync
from aftersalesagent.harness.journal import EventConflict, Journal
from aftersalesagent.local_config import options_from_settings
from aftersalesagent.models.base import ModelBackend
from aftersalesagent.models.deepseek import DeepSeekBackend
from aftersalesagent.models.scenarios import ScenarioName
from aftersalesagent.policies.application import load_application_policy
from aftersalesagent.schemas.contracts import (
    ConfirmationEvent,
    MessageEvent,
    RunLimits,
    TurnResult,
)
from aftersalesagent.skills.registry import SkillRegistry

ROOT = Path(__file__).resolve().parents[3]
bearer = HTTPBearer(auto_error=False)


class CreateSession(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scenario: ScenarioName = "cancel"
    limits: RunLimits | None = None


class SendMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    event_id: str = Field(min_length=1, max_length=128)
    content: str = Field(min_length=1, max_length=4000)


class ConfirmAction(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    event_id: str = Field(min_length=1, max_length=128)
    approved: bool


def create_app(
    *,
    data_dir: Path | None = None,
    template: Path | None = None,
    backend_factory: Callable[[str], ModelBackend] | None = None,
    settings: Settings | None = None,
    api_key: SecretStr | None = None,
    enable_paid_api: bool = False,
    knowledge_backend: ModelBackend | None = None,
) -> FastAPI:
    settings = settings or Settings.from_environment()
    if settings.runtime != "langgraph" or settings.backend not in {"mock", "deepseek"}:
        raise ValueError("Use langgraph and mock/deepseek backend")
    if settings.backend == "deepseek":
        if not enable_paid_api:
            raise ValueError("DeepSeek mode requires explicit --enable-paid-api")
        options = options_from_settings(settings)
        if backend_factory is None:
            if api_key is None or not api_key.get_secret_value():
                raise ValueError("Configure DEEPSEEK_API_KEY locally")
            secret = api_key

            def live_backend_factory(_scenario):
                return DeepSeekBackend(options, secret, allow_network=True)

            backend_factory = live_backend_factory

    directory = (data_dir or ROOT / ".local").resolve()
    source = (template or directory / "databases" / "retail_template.sqlite").resolve()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        directory.mkdir(parents=True, exist_ok=True)
        if not source.exists():
            store = await run_sync(lambda: import_pinned_database(ROOT, source))
            store.close()
        policy = load_application_policy(ROOT)
        skill_registry = None
        if settings.skill_id is not None:
            skill_registry = SkillRegistry(directory / "skills/registry.sqlite")
            selected_skill_id = settings.skill_id
            bundle = ROOT / "skills" / selected_skill_id / "1.0.0"
            installed = await skill_registry.list_metadata()
            if bundle.is_dir() and not any(
                m.skill_id == selected_skill_id and m.version == "1.0.0" for m in installed
            ):
                await run_sync(
                    lambda: skill_registry.install(ROOT / "skills", selected_skill_id, "1.0.0")
                )
            # released 必须先人工激活；每个新会话仍重新解析，不能在此缓存。
            await skill_registry.load(settings.skill_id, settings.skill_version)
        journal = Journal(directory / "application.sqlite")
        async with AsyncSqliteSaver.from_conn_string(
            str(directory / "checkpoints.sqlite")
        ) as saver:
            await saver.setup()
            hub = SessionHub(
                source,
                directory,
                policy,
                journal,
                saver,
                backend_factory,
                backend_name=settings.backend,
                tool_transport=settings.tool_transport,
                skill_provider=skill_registry,
                skill_id=settings.skill_id,
                skill_version=settings.skill_version,
                model_name=settings.model_name,
                temperature=settings.model_temperature,
                max_output_tokens=settings.model_max_output_tokens,
            )
            app.state.hub = hub
            try:
                yield
            finally:
                if getattr(app.state, "knowledge", None) is not None:
                    await app.state.knowledge.close()
                await hub.close()
                journal.close()

    app = FastAPI(title="AfterSalesAgent", version="0.3.0", lifespan=lifespan)

    app.include_router(
        knowledge_router(
            ROOT,
            directory,
            source,
            settings,
            api_key,
            enable_paid_api,
            app.state,
            knowledge_backend,
        )
    )

    @app.get("/knowledge.js", include_in_schema=False)
    async def knowledge_javascript():
        return FileResponse(ROOT / "web/knowledge.js", media_type="text/javascript")

    @app.exception_handler(EventConflict)
    async def conflict_handler(_request: Request, exc: EventConflict):
        return JSONResponse(
            status_code=409,
            content={"error": {"code": "event_conflict", "message": str(exc), "retryable": False}},
        )

    @app.exception_handler(Exception)
    async def infrastructure_handler(_request: Request, _exc: Exception):
        # 不把数据库路径、SQL、鉴权信息或异常原文返回客户端。
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "code": "infrastructure_error",
                    "message": "Service unavailable. Retry using the original event_id.",
                    "retryable": True,
                }
            },
        )

    async def authorized(
        sid: str,
        request: Request,
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    ):
        hub: SessionHub = request.app.state.hub
        entry = await run_sync(lambda: hub.journal.session(sid))
        digest = hashlib.sha256(
            (credentials.credentials if credentials else "").encode()
        ).hexdigest()
        if (
            entry is None
            or credentials is None
            or not secrets.compare_digest(digest, entry["token_hash"])
        ):
            raise HTTPException(status_code=404, detail="Session not found or unavailable")
        return await hub.get(sid)

    runtime_dependency = Depends(authorized)

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "stage": "M3",
            "backend": "offline_scenario_mock" if settings.backend == "mock" else "deepseek",
            "model": settings.model_name,
            "paid_api_enabled": enable_paid_api,
            "tool_transport": settings.tool_transport,
            "knowledge_transport": settings.knowledge_transport,
            "skill_id": settings.skill_id,
            "skill_selector": settings.skill_version,
        }

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(ROOT / "web/index.html")

    @app.get("/app.js", include_in_schema=False)
    async def javascript():
        return FileResponse(ROOT / "web/app.js", media_type="text/javascript")

    @app.get("/styles.css", include_in_schema=False)
    async def styles():
        return FileResponse(ROOT / "web/styles.css", media_type="text/css")

    @app.post("/sessions", status_code=201)
    async def create(body: CreateSession, request: Request):
        return await request.app.state.hub.create(
            body.scenario, body.limits or settings.run_limits()
        )

    @app.get("/sessions/{sid}")
    async def get_session(sid: str, runtime=runtime_dependency):
        return {
            "session": await runtime.get_state(sid),
            "usage": await run_sync(lambda: runtime.journal.usage(sid)),
            "latest_request": await run_sync(lambda: runtime.journal.latest(sid)),
        }

    @app.post("/sessions/{sid}/messages", response_model=TurnResult)
    async def message(sid: str, body: SendMessage, runtime=runtime_dependency):
        return await runtime.advance(
            sid, MessageEvent(event_id=body.event_id, content=body.content)
        )

    @app.post("/sessions/{sid}/actions/{action_id}/confirm", response_model=TurnResult)
    async def confirm(sid: str, action_id: str, body: ConfirmAction, runtime=runtime_dependency):
        return await runtime.advance(
            sid,
            ConfirmationEvent(event_id=body.event_id, action_id=action_id, approved=body.approved),
        )

    @app.get("/sessions/{sid}/orders")
    async def visible_orders(sid: str, runtime=runtime_dependency):
        return {"orders": await run_sync(lambda: runtime.service.visible_orders(sid))}

    @app.get("/sessions/{sid}/trace")
    async def trace(
        sid: str,
        after: int = Query(default=-1, ge=-1),
        output: str = Query(default="json", pattern="^(json|jsonl)$"),
        runtime=runtime_dependency,
    ):
        # 回放端点只读账本。重建JSONL属于文件投影，不重新执行图。
        if runtime.trace_directory:
            await run_sync(lambda: runtime.journal.export(sid, runtime.trace_directory))
        events = await run_sync(lambda: runtime.journal.events(sid, after))
        if output == "jsonl":
            return PlainTextResponse(
                "".join(event.model_dump_json() + "\n" for event in events),
                media_type="application/x-ndjson",
            )
        return {"events": events}

    return app
