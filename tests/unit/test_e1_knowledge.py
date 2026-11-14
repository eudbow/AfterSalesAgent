"""来源/日期/分词/引用的反例；不进行真实模型请求。"""

import asyncio
import json
from pathlib import Path

import pytest

from aftersalesagent.config import Settings
from aftersalesagent.evaluation.e1_runner import verify
from aftersalesagent.evaluation.e1_suite import load
from aftersalesagent.knowledge.corpus import eligible, load_corpus
from aftersalesagent.knowledge.factory import make_service
from aftersalesagent.knowledge.retrieval import SQLiteKnowledgeIndex
from aftersalesagent.knowledge.service import KnowledgeQuestion, RequestConflict
from aftersalesagent.schemas.contracts import ModelResponse, ToolCall

ROOT = Path(__file__).resolve().parents[2]


class CounterBackend:
    def __init__(self, body=None, known=True, tool=False):
        self.body = body
        self.known, self.tool = known, tool
        self.seen = []

    async def generate(self, request):
        self.seen.append(request)
        chunks = json.loads(request.messages[-1].content)["evidence"]
        target = next((c for c in chunks if c["chunk_id"] == "R01"), None)
        body = self.body or {
            "answer": "普通顾客60天，会员365天。",
            "status": "answered",
            "citations": ["R01"],
            "policy_versions": [target["version"]] if target else [],
        }
        return ModelResponse(
            content=json.dumps(body, ensure_ascii=False),
            input_tokens=10 if self.known else None,
            output_tokens=5 if self.known else None,
            tool_calls=[ToolCall(call_id="bad", name="cancel_pending_order", arguments={})]
            if self.tool
            else [],
        )


def service(tmp_path, template, backend, **kwargs):
    return make_service(
        ROOT,
        Settings(),
        None,
        template=template,
        database=tmp_path / "knowledge.sqlite",
        backend_override=backend,
        **kwargs,
    )


def test_official_spans_and_unknown_dates_are_real_snapshots():
    chunks, manifest = load_corpus(ROOT)
    assert len(chunks) == 23 and manifest["official_documents"] == 2
    official = [c for c in chunks if c.origin == "official"]
    assert len(official) == 17
    assert all(c.effective_from is None and c.effective_until is None for c in official)
    assert {c.scope for c in official} == {"ikea_cn"}
    for c in chunks:
        raw = (ROOT / c.source_text_path).read_text("utf-8")
        assert raw[c.source_start : c.source_end]
    for split in ("dev", "validation", "test"):
        cases, _ = load(ROOT, split)
        assert len(cases) == 12
        assert all("required_ids" not in c["public"] for c in cases)


@pytest.mark.parametrize(
    "day,expected", [("2026-06-30", {"D01", "D03"}), ("2026-07-01", {"D02", "D04"})]
)
def test_date_boundary_and_merchant_scope_filter(day, expected):
    chunks, _ = load_corpus(ROOT)
    assert {c.chunk_id for c in eligible(chunks, {"scope": "demo_store", "as_of": day})} == expected
    assert eligible(chunks, {"scope": "ikea_cn", "as_of": day}) == []
    assert all(c.scope == "ikea_cn" for c in eligible(chunks, {"scope": "ikea_cn"}))


def test_unsupported_filters_do_not_silently_cross_scopes():
    chunks, _ = load_corpus(ROOT)
    with pytest.raises(ValueError):
        eligible(chunks, {"scope": "other_merchant"})
    with pytest.raises(ValueError):
        eligible(chunks, {"purchase_date": "2020-01-01"})


def test_actual_chinese_two_character_query_and_sql_injection_are_safe():
    chunks, _ = load_corpus(ROOT)
    index = SQLiteKnowledgeIndex(chunks, "char")
    assert "R02" in {c.chunk_id for c in index.retrieve("凭证", {"scope": "ikea_cn"}, 5)}
    index.retrieve('凭证 " OR 1=1; DROP TABLE evidence_fts;', {"scope": "ikea_cn"}, 5)
    assert index.connection.execute("SELECT count(*) FROM evidence_fts").fetchone()[0] == 23
    evidence = asyncio.run(index.search("银行卡凭证", {"scope": "ikea_cn"}, 5))
    assert evidence and evidence[0].document_id == "ikea_returns"
    assert evidence[0].chunk_id and evidence[0].score < 0
    index.close()


def test_dev_recall_uses_gold_only_in_evaluator():
    chunks, _ = load_corpus(ROOT)
    index = SQLiteKnowledgeIndex(chunks, "char")
    cases, _ = load(ROOT, "dev")
    for case in cases:
        p, c = case["public"], case["criteria"]
        found = {x.chunk_id for x in index.retrieve(p["question"], p["filters"], 5)}
        assert set(c["required_ids"]) <= found
    index.close()


def test_full_and_rag_have_identical_nonknowledge_inputs_and_no_hidden_answers(
    tmp_path, retail_template
):
    backend = CounterBackend()

    async def execute():
        app = service(tmp_path, retail_template, backend)
        full = await app.ask(
            KnowledgeQuestion(question="普通顾客与会员退货期限？", mode="full"), "full"
        )
        rag = await app.ask(
            KnowledgeQuestion(question="普通顾客与会员退货期限？", mode="rag"), "rag"
        )
        assert len(full["context_ids"]) == 17 and len(rag["context_ids"]) <= 5
        assert full["context_characters"] > rag["context_characters"]
        a, b = backend.seen
        assert a.messages[:-1] == b.messages[:-1] and a.tools == b.tools
        assert a.tool_choice == b.tool_choice == "none" and a.temperature == b.temperature
        for request in backend.seen:
            payload = json.loads(request.messages[-1].content)
            assert set(payload) == {"question", "scope", "as_of", "evidence"}
            assert "criteria" not in payload and "fact_patterns" not in payload
        assert full["citation_versions_valid"] and rag["citation_ids_valid"]
        await app.close()

    asyncio.run(execute())


def test_request_idempotence_is_durable_and_conflicts_never_call_model(tmp_path, retail_template):
    first = CounterBackend()

    async def execute():
        app = service(tmp_path, retail_template, first)
        q = KnowledgeQuestion(question="普通顾客退货期限？")
        original = await app.ask(q, "same")
        assert await app.ask(q, "same") == original and len(first.seen) == 1
        with pytest.raises(RequestConflict):
            await app.ask(KnowledgeQuestion(question="另外的问题"), "same")
        await app.close()
        second = CounterBackend()
        restored = service(tmp_path, retail_template, second)
        assert await restored.ask(q, "same") == original and not second.seen
        await restored.close()

    asyncio.run(execute())


@pytest.mark.parametrize(
    "body",
    [
        {
            "answer": "凭空回答",
            "status": "answered",
            "citations": ["FAKE"],
            "policy_versions": ["fake"],
        },
        {"answer": "没有依据", "status": "answered", "citations": [], "policy_versions": []},
        {
            "answer": "错版本",
            "status": "answered",
            "citations": ["R01"],
            "policy_versions": ["wrong"],
        },
    ],
)
def test_unverified_answer_is_preserved_for_audit_but_hidden_from_ui(
    body, tmp_path, retail_template
):
    async def execute():
        app = service(tmp_path, retail_template, CounterBackend(body=body))
        result = await app.ask(KnowledgeQuestion(question="退货期限"), "bad")
        public = app.public_result(result)
        assert public["answer"]["status"] == "insufficient_evidence"
        assert public["sources"] == [] and "model_request" not in public
        assert result["answer"]["answer"] == body["answer"]
        await app.close()

    asyncio.run(execute())


def test_unexpected_model_tool_call_is_never_executed(tmp_path, retail_template, monkeypatch):
    from aftersalesagent.business.service import RetailService

    def forbidden(*args, **kwargs):
        raise AssertionError("Knowledge QA must never dispatch a business action")

    monkeypatch.setattr(RetailService, "dispatch", forbidden)

    async def execute():
        app = service(tmp_path, retail_template, CounterBackend(tool=True))
        result = await app.ask(KnowledgeQuestion(question="退货期限"), "tool")
        assert result["error"]["code"] == "knowledge_unexpected_tool_or_truncation"
        await app.close()

    asyncio.run(execute())


def test_missing_usage_remains_unknown(tmp_path, retail_template):
    async def execute():
        app = service(tmp_path, retail_template, CounterBackend(known=False))
        result = await app.ask(KnowledgeQuestion(question="退货期限"), "unknown")
        assert result["usage"]["reported_tokens"] is None
        assert result["usage"]["unknown_usage_calls"] == 1 and result["cost_estimate"] is None
        await app.close()

    asyncio.run(execute())


def test_key_facts_with_wrong_citation_or_version_cannot_pass():
    cases, _ = load(ROOT, "dev")
    criteria = cases[0]["criteria"]
    result = {
        "answer": {
            "answer": "60天和365天",
            "status": "answered",
            "citations": ["R02"],
            "policy_versions": criteria["versions"],
        },
        "error": None,
        "citation_ids_valid": True,
        "citation_versions_valid": True,
        "context_ids": ["R01", "R02"],
    }
    verdict = verify(result, criteria)
    assert verdict["checks"]["key_facts"] and not verdict["success"]
    assert verdict["evidence_recall"] == 1


def test_required_evidence_coverage_does_not_reject_related_extra_citations():
    cases, _ = load(ROOT, "dev")
    criteria = cases[0]["criteria"]
    result = {
        "answer": {
            "answer": "60天和365天",
            "status": "answered",
            "citations": ["R01", "R02"],
            "policy_versions": criteria["versions"],
        },
        "error": None,
        "citation_ids_valid": True,
        "citation_versions_valid": True,
        "context_ids": ["R01", "R02"],
    }
    verdict = verify(result, criteria)
    assert verdict["success"]
    assert not verdict["strict_reference_set_exact"]
    assert verdict["required_reference_precision"] == 0.5
