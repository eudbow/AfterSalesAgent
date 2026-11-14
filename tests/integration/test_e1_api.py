"""实际FastAPI知识入口，无Key、无网络、无业务写操作。"""

import asyncio
import json

import httpx

from aftersalesagent.api.app import create_app
from aftersalesagent.schemas.contracts import ModelResponse


class Backend:
    def __init__(self):
        self.calls = 0

    async def generate(self, request):
        self.calls += 1
        evidence = json.loads(request.messages[-1].content)["evidence"]
        chunk = next(c for c in evidence if c["chunk_id"] == "R02")
        return ModelResponse(
            content=json.dumps(
                {
                    "answer": "需原始购物凭证、发票、银行卡和原始签购单。",
                    "status": "answered",
                    "citations": ["R02"],
                    "policy_versions": [chunk["version"]],
                },
                ensure_ascii=False,
            ),
            input_tokens=10,
            output_tokens=5,
        )


def test_knowledge_api_cache_sources_and_safe_public_response(tmp_path, retail_template):
    backend = Backend()
    app = create_app(data_dir=tmp_path, template=retail_template, knowledge_backend=backend)

    async def execute():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://local"
            ) as client:
                health = await client.get("/knowledge/status")
                assert health.json()["stage"] == "E1" and health.json()["corpus_ready"]
                body = {"request_id": "query-1", "question": "银行卡退货凭证材料？", "mode": "rag"}
                a = await client.post("/knowledge/ask", json=body)
                b = await client.post("/knowledge/ask", json=body)
                assert a.status_code == b.status_code == 200 and a.json() == b.json()
                result = a.json()
                assert backend.calls == 1 and result["sources"][0]["chunk_id"] == "R02"
                assert result["sources"][0]["effective_from"] is None
                assert "model_request" not in result and "model_response" not in result
                conflict = await client.post(
                    "/knowledge/ask", json={**body, "question": "不同的问题"}
                )
                assert conflict.status_code == 409 and backend.calls == 1
                script = await client.get("/knowledge.js")
                assert script.status_code == 200 and "textContent" in script.text

    asyncio.run(execute())
