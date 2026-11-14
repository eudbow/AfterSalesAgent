"""E4 integration with a real MCP child and HTTP Mock; no model downloads."""

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from aftersalesagent.api.app import create_app
from aftersalesagent.config import Settings
from aftersalesagent.knowledge.corpus import load_corpus
from aftersalesagent.knowledge.e4 import E4Config, build_index
from aftersalesagent.mcp_transport.adapters import MCPKnowledgeProvider

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/evaluation/e4_bm25_v1.json"


def test_real_e4_mcp_child_matches_native_scopes_dates_and_order():
    config = E4Config.model_validate_json(CONFIG.read_text("utf-8"))
    index = build_index(ROOT, config)
    remote = MCPKnowledgeProvider(index, knowledge_config=CONFIG)

    async def execute():
        try:
            cases = [
                ("SKYTTA 胶带 地板", {"scope": "ikea_cn"}),
                ("配件补寄 运费", {"scope": "demo_store", "as_of": "2026-06-30"}),
                ("配件补寄 运费", {"scope": "demo_store", "as_of": "2026-07-01"}),
                ("退货期限", {"scope": "demo_conflict", "category": "e4_return"}),
                ("床垫", {"scope": "ikea_cn", "as_of": "2010-01-01"}),
            ]
            for query, filters in cases:
                expected = await index.provide(query, filters, 5)
                actual = await remote.provide(query, filters, 5)
                assert actual == expected
                assert remote.last_trace is not None
                assert remote.last_trace["selected_ids"] == [c.chunk_id for c in expected]
            assert remote.remote.protocol_version
        finally:
            await remote.aclose()
            remote.close()

    asyncio.run(execute())


def test_e4_http_mock_idempotence_citations_and_old_business_still_available(
    tmp_path, retail_template
):
    class Backend:
        calls = 0

        async def generate(self, request):
            from aftersalesagent.schemas.contracts import ModelResponse

            self.calls += 1
            assert request.tool_choice == "none"
            payload = json.loads(request.messages[-1].content)
            assert "support_groups" not in payload and "semantic_checklist" not in payload
            evidence = payload["evidence"]
            item = evidence[0]
            return ModelResponse(
                content=json.dumps(
                    {
                        "answer": item["text"],
                        "status": "answered",
                        "citations": [item["chunk_id"]],
                        "policy_versions": [item["version"]],
                    },
                    ensure_ascii=False,
                ),
                input_tokens=10,
                output_tokens=10,
            )

    backend = Backend()
    settings = Settings(knowledge_config="configs/evaluation/e4_bm25_v1.json")
    app = create_app(
        data_dir=tmp_path, template=retail_template, settings=settings, knowledge_backend=backend
    )

    async def execute():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://local"
            ) as client:
                assert (await client.get("/health")).status_code == 200
                assert (await client.get("/knowledge/status")).json()["stage"] == "E4"
                question = {
                    "request_id": "e4-http-1",
                    "question": "SKYTTA 胶带 地板",
                    "mode": "rag",
                }
                first = await client.post("/knowledge/ask", json=question)
                repeat = await client.post("/knowledge/ask", json=question)
                assert first.status_code == 200 and first.json() == repeat.json()
                assert backend.calls == 1 and first.json()["citation_ids_valid"]
                assert first.json()["retrieval_trace"]["method"] == "bm25"
                assert "model_request" not in first.json()
                conflict = await client.post(
                    "/knowledge/ask", json={**question, "question": "不同问题"}
                )
                assert conflict.status_code == 409

    asyncio.run(execute())


def test_material_split_families_and_evidence_are_sealed():
    from aftersalesagent.evaluation.e4.suite import load

    families = {}
    chunks, _ = load_corpus(ROOT, "e4_v1")
    by_id = {c.chunk_id for c in chunks}
    for split, count in (("dev", 60), ("validation", 45), ("test", 45)):
        cases, _ = load(ROOT, split)
        assert len(cases) == count
        for case in cases:
            p, c = case["public"], case["criteria"]
            assert set(p) == {"task_id", "question", "filters"}
            assert families.setdefault(c["family_id"], split) == split
            assert all(set(group) <= by_id for group in c["support_groups"])
    assert len(families) == 50


def test_gold_v2_keeps_public_inputs_and_adds_only_supported_paths():
    from aftersalesagent.evaluation.e4.suite import load

    for split in ("dev", "validation", "test"):
        before, _ = load(ROOT, split, "e4_v1")
        after, _ = load(ROOT, split, "e4_v2")
        assert [c["public"] for c in before] == [c["public"] for c in after]
        for first, second in zip(before, after, strict=True):
            original = {tuple(group) for group in first["criteria"]["support_groups"]}
            revised = {tuple(group) for group in second["criteria"]["support_groups"]}
            assert original <= revised
            assert first["criteria"]["expected_status"] == second["criteria"]["expected_status"]
            assert second["criteria"]["secondary_tags"]


def test_unknown_model_or_corpus_configuration_fails():
    with pytest.raises(ValueError):
        E4Config.model_validate({"model_id": "BAAI/bge-m3"})
    with pytest.raises(ValueError):
        E4Config.model_validate({"corpus_version": "untrusted"})
