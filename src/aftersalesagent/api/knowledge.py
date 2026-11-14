"""知识咨询路由：与订单执行分开，使用同一模型配置；按请求ID去重。"""

import asyncio
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import Field

from aftersalesagent.knowledge.factory import make_service
from aftersalesagent.knowledge.service import KnowledgeQuestion, RequestConflict


class AskKnowledge(KnowledgeQuestion):
    request_id: str = Field(min_length=1, max_length=128)


def router(
    root: Path,
    directory: Path,
    template: Path,
    settings,
    secret,
    enable_paid_api,
    state,
    backend_override=None,
):
    result = APIRouter()
    init_lock = asyncio.Lock()

    @result.get("/knowledge/status")
    async def status():
        return {
            "stage": "E4" if settings.knowledge_config else "E1",
            "backend": "deepseek" if settings.backend == "deepseek" else "mock_demo",
            "corpus_ready": (
                root
                / (
                    "data/knowledge/e4_v1/manifest.json"
                    if settings.knowledge_config
                    else "data/knowledge/e1_v1/manifest.json"
                )
            ).exists(),
            "knowledge_config": settings.knowledge_config,
            "key_tested": False,
            "transport": settings.knowledge_transport,
        }

    @result.post("/knowledge/ask")
    async def ask(body: AskKnowledge):
        async with init_lock:
            if getattr(state, "knowledge", None) is None:
                try:
                    state.knowledge = make_service(
                        root,
                        settings,
                        secret,
                        template=template,
                        database=directory
                        / (
                            "knowledge/e4.sqlite"
                            if settings.knowledge_config
                            else "knowledge/e1.sqlite"
                        ),
                        enable_paid_api=enable_paid_api,
                        backend_override=backend_override,
                    )
                except ValueError:
                    raise HTTPException(
                        status_code=503, detail="Knowledge configuration unavailable"
                    ) from None
        question = KnowledgeQuestion.model_validate(body.model_dump(exclude={"request_id"}))
        try:
            response = await state.knowledge.ask(question, body.request_id)
        except RequestConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None
        except ValueError:
            raise HTTPException(
                status_code=400, detail="Invalid knowledge query or context limit"
            ) from None
        return state.knowledge.public_result(response)

    return result
