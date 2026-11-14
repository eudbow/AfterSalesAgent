"""E4 counterexamples for filtering, fusion, provenance, ledger and cache validation."""

import asyncio
import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from aftersalesagent.knowledge.corpus import KnowledgeChunk, load_corpus
from aftersalesagent.knowledge.e4 import (
    E4Config,
    E4Index,
    build_index,
    check_vectors,
    vector_signature,
)
from aftersalesagent.knowledge.service import KnowledgeQuestion, KnowledgeService, RequestConflict
from aftersalesagent.mcp_transport.adapters import MCPKnowledgeProvider
from aftersalesagent.schemas.contracts import ModelResponse

ROOT = Path(__file__).resolve().parents[2]


def chunk(cid, text="凭证材料", **kwargs):
    return KnowledgeChunk(
        chunk_id=cid,
        document_id="doc-" + cid,
        section="条款 " + cid,
        text=text,
        version="v1",
        source="https://example.test/policy",
        scope="ikea_cn",
        origin="official",
        source_text_path="source.txt",
        source_start=0,
        source_end=len(text),
        source_excerpt_sha256=hashlib.sha256(text.encode()).hexdigest(),
        **kwargs,
    )


def unit(axis):
    value = [0.0] * 512
    value[axis] = 1.0
    return value


class Model:
    def __init__(self):
        self.queries = []
        self.pairs = []

    def encode(self, texts, *, kind):
        self.queries.append((texts, kind))
        return [unit(0) for _ in texts]

    def rerank(self, query, chunks):
        self.pairs.append([c.chunk_id for c in chunks])
        return [100.0 if c.chunk_id == "B" else -100 for c in chunks]


def test_dense_filters_before_top_k_and_stable_ties():
    chunks = [chunk("A"), chunk("B"), chunk("X").model_copy(update={"scope": "demo_store"})]
    model = Model()
    index = E4Index(chunks, E4Config(method="dense"), model=model, vectors=[unit(0)] * 3)
    assert [c.chunk_id for c, _ in index.ranked("材料", {"scope": "ikea_cn"}, 1)] == ["A"]
    assert [c.chunk_id for c, _ in index.ranked("材料", {"scope": "demo_store"}, 1)] == ["X"]
    assert index.ranked("材料", {"scope": "ikea_cn", "as_of": "2026-01-01"}, 5) == []
    assert model.queries == [(["材料"], "query")] * 2
    with pytest.raises(ValueError):
        index.ranked("材料", {"scope": "other"}, 5)
    index.close()


def test_rrf_uses_ranks_and_reranker_only_sees_filtered_fusion_pool():
    chunks = [chunk("A", "凭证凭证凭证材料"), chunk("B", "维修服务"), chunk("C", "清洁保养")]
    model = Model()
    index = E4Index(
        chunks,
        E4Config(method="rerank", candidate_k=1, fusion_k=2),
        model=model,
        vectors=[unit(1), unit(0), unit(2)],
    )
    ranked = index.ranked("凭证", {"scope": "ikea_cn"}, 2)
    assert index.last_trace is not None
    assert [c.chunk_id for c, _ in ranked] == ["B", "A"]
    fused = index.last_trace["stages"]["fusion"]
    assert {item["score"] for item in fused} == {1 / 61}
    assert set(model.pairs[0]) == {"A", "B"}
    assert "C" not in model.pairs[0]
    assert index.last_trace["stages"]["bm25"][0]["direction"] == "lower"
    index.close()


@pytest.mark.parametrize("vectors", [[[float("nan")] * 512], [[0.0] * 512], [[1.0]], []])
def test_bad_vectors_are_rejected(vectors):
    with pytest.raises(ValueError):
        check_vectors(vectors, 1)


def test_vector_signature_changes_with_text_precision_revision():
    chunks = [chunk("A")]
    config = E4Config(method="dense", embedding_revision="rev-a")
    signature = vector_signature(chunks, config)
    assert signature != vector_signature(chunks, config.model_copy(update={"precision": "fp16"}))
    assert signature != vector_signature(
        chunks, config.model_copy(update={"embedding_revision": "rev-b"})
    )
    assert signature != vector_signature([chunk("A", "新内容")], config)


def test_index_fingerprint_binds_actual_vector_contents():
    config = E4Config(method="dense")
    first = E4Index([chunk("A")], config, model=Model(), vectors=[unit(0)])
    second = E4Index([chunk("A")], config, model=Model(), vectors=[unit(1)])
    assert first.index_fingerprint != second.index_fingerprint
    first.close()
    second.close()


def test_real_e4_corpus_has_traceable_spans_without_navigation_or_us_clause():
    chunks, manifest = load_corpus(ROOT, "e4_v1")
    assert 100 <= len(chunks) <= 300
    assert manifest["official_documents"] == 10
    for c in chunks:
        text = (ROOT / c.source_text_path).read_text("utf-8")
        assert c.text == text[c.source_start : c.source_end]
        assert "美国地区" not in c.text and "登录宜家账号" not in c.text


def test_bm25_e4_keeps_original_algorithm():
    chunks, _ = load_corpus(ROOT, "e4_v1")
    index = E4Index(chunks, E4Config())
    assert index.ranked("SKYTTA 滑门 胶带 地板", {"scope": "ikea_cn"}, 5) == index.lexical.ranked(
        "SKYTTA 滑门 胶带 地板", {"scope": "ikea_cn"}, 5
    )
    index.close()


def test_new_service_ledger_is_durable_without_retriever_sqlite_contract(tmp_path):
    class Backend:
        calls = 0

        async def generate(self, request):
            self.calls += 1
            return ModelResponse(
                content=json.dumps(
                    {
                        "answer": "请提供凭证",
                        "status": "answered",
                        "citations": ["A"],
                        "policy_versions": ["v1"],
                    }
                ),
                input_tokens=10,
                output_tokens=5,
            )

    backend = Backend()
    path = tmp_path / "ledger.sqlite"

    def service():
        index = E4Index([chunk("A")], E4Config())
        assert not hasattr(index, "connection")
        return KnowledgeService(
            index,
            backend,
            core_policy="read only",
            tools=[],
            model_name=None,
            ledger=sqlite3.connect(path),
            close_ledger=True,
        )

    async def run():
        first = service()
        question = KnowledgeQuestion(question="凭证材料？")
        a = await first.ask(question, "same")
        assert a["timings"]["end_to_end_seconds"] >= a["timings"]["retrieval_seconds"]
        await first.close()
        second = service()
        assert await second.ask(question, "same") == a
        with pytest.raises(RequestConflict):
            await second.ask(KnowledgeQuestion(question="另一个问题"), "same")
        assert backend.calls == 1
        await second.close()

    asyncio.run(run())


def test_mcp_rejects_config_drift_and_duplicate_evidence():
    index = E4Index([chunk("A")], E4Config())
    provider = MCPKnowledgeProvider(index, knowledge_config=Path("config.json"))

    class Remote:
        async def call(self, *args):
            return {"index_fingerprint": "wrong", "evidence": []}

    provider.remote = Remote()  # type: ignore[assignment] - injectable protocol fault fixture
    with pytest.raises(ValueError, match="configuration"):
        asyncio.run(provider.provide("凭证", {}, 5))

    class Duplicate:
        async def call(self, *args):
            return {
                "index_fingerprint": index.index_fingerprint,
                "evidence": [{"chunk": chunk("A").model_dump(), "score": -1}] * 2,
            }

    provider.remote = Duplicate()  # type: ignore[assignment] - injectable protocol fault fixture
    with pytest.raises(ValueError, match="duplicate"):
        asyncio.run(provider.provide("凭证", {}, 5))
    index.close()


def test_missing_or_wrong_model_never_downloads(tmp_path):
    (tmp_path / "data/knowledge/e4_v1").mkdir(parents=True)
    (tmp_path / "data/knowledge/e4_v1/manifest.json").write_text('{"files": {}}')
    (tmp_path / "data/knowledge/e4_v1/chunks.jsonl").write_text("")
    with pytest.raises(FileNotFoundError):
        build_index(tmp_path, E4Config(method="dense"))
