"""C3 material, comparison, gate and review contracts; no network/model downloads."""

import json
from collections import Counter
from pathlib import Path

import pytest

from aftersalesagent.evaluation.e4.execution import sha
from aftersalesagent.evaluation.e4_c3 import (
    OPTIONS,
    C3Config,
    automatic,
    call_plan,
    input_size,
    load_materials,
    make_request,
    stage_allowed,
    tool_specs,
    worker_config,
)
from aftersalesagent.knowledge.corpus import eligible
from aftersalesagent.knowledge.e4 import E4Index

ROOT = Path(__file__).resolve().parents[2]


def test_c3_fixed_material_counts_family_isolation_and_source_spans():
    families = []
    counts = Counter()
    for split in ("dev", "validation", "test"):
        chunks, cases, _ = load_materials(ROOT, split)
        assert len(chunks) == 168 and len(cases) == 30
        local = Counter(c["criteria"]["family_id"] for c in cases)
        assert len(local) == 15 and set(local.values()) == {2}
        families += list(local)
        counts.update(c["criteria"]["challenge"] for c in cases)
        by_id = {c.chunk_id: c for c in chunks}
        for case in cases:
            allowed = {c.chunk_id for c in eligible(chunks, case["public"]["filters"])}
            assert all(set(group) <= allowed for group in case["criteria"]["support_groups"])
            assert all(
                cid in by_id for group in case["criteria"]["support_groups"] for cid in group
            )
    assert len(set(families)) == 45
    assert counts == {
        "identifier": 18,
        "paraphrase": 18,
        "condition": 18,
        "multi_evidence": 12,
        "no_answer": 12,
        "date_scope": 6,
        "conflict": 6,
    }


def test_kitchen_appliance_heading_no_longer_carries_metod_duration():
    chunks, _, _ = load_materials(ROOT, "dev")
    appliance = next(c for c in chunks if c.chunk_id == "C3-kitchen-022")
    assert "25年" not in appliance.section and "家电" in appliance.section
    assert "五(5)年" in appliance.text
    assert all(c.scope == "ikea_cn" for c in chunks if c.origin == "official")


def test_full_and_rag_change_only_evidence_in_model_request():
    chunks, cases, _ = load_materials(ROOT, "dev")
    case = cases[0]["public"]
    tools = tool_specs(ROOT)
    full = make_request(ROOT, case, eligible(chunks, case["filters"]), tools, OPTIONS)
    rag = make_request(ROOT, case, chunks[:5], tools, OPTIONS)
    assert full.messages[:2] == rag.messages[:2] and full.tools == rag.tools
    assert full.tool_choice == rag.tool_choice == "none"
    left = json.loads(full.messages[-1].content)
    right = json.loads(rag.messages[-1].content)
    assert len(left.pop("evidence")) == 156 and len(right.pop("evidence")) == 5
    assert left == right
    assert (
        "semantic_checklist" not in full.model_dump_json()
        and "support_groups" not in full.model_dump_json()
    )
    assert input_size(full, OPTIONS)["serialized_characters"] < 200000


def test_exact_call_plan_reuses_smoke_and_bounds_repeated_diagnostics():
    _, cases, _ = load_materials(ROOT, "dev")
    full = call_plan(cases, "dev")
    smoke = call_plan(cases, "dev", True)

    def key(item):
        return item[0]["public"]["task_id"], item[1], item[2]

    assert len(full) == 120 and len(smoke) == 18
    assert len(set(map(key, full))) == 120
    assert set(map(key, smoke)) <= set(map(key, full))
    for split in ("validation", "test"):
        _, other, _ = load_materials(ROOT, split)
        assert len(call_plan(other, split)) == 90


def test_c3_validity_gate_is_distinct_from_old_failed_effect_gate(tmp_path):
    stage_allowed(tmp_path, "dev", "p")
    folder = tmp_path / ".local/e4_c3_artifacts"
    folder.mkdir(parents=True)
    file = folder / "dev_validity_gate.json"
    file.write_text(json.dumps({"execution_valid": False, "protocol_sha256": "p"}), "utf-8")
    with pytest.raises(ValueError):
        stage_allowed(tmp_path, "validation", "p")
    file.write_text(json.dumps({"execution_valid": True, "protocol_sha256": "different"}), "utf-8")
    with pytest.raises(ValueError):
        stage_allowed(tmp_path, "validation", "p")
    file.write_text(
        json.dumps({"execution_valid": True, "protocol_sha256": "p", "quality_improvement": False}),
        "utf-8",
    )
    stage_allowed(tmp_path, "validation", "p")


def test_regex_is_not_a_hidden_c3_hard_gate():
    _, cases, _ = load_materials(ROOT, "dev")
    assert all(
        not c["criteria"]["required_patterns"] and not c["criteria"]["forbidden_patterns"]
        for c in cases
    )
    no_answer = next(
        c["criteria"] for c in cases if c["criteria"]["expected_status"] == "insufficient_evidence"
    )
    result = {
        "answer": {"status": "insufficient_evidence", "answer": "无法从资料确定", "citations": []},
        "error": None,
        "citation_ids_valid": True,
        "citation_versions_valid": True,
    }
    assert automatic(result, no_answer)["automatic_pass"]


def test_c3_lexical_path_and_worker_adapter_preserve_explicit_corpus():
    chunks, _, _ = load_materials(ROOT, "dev")
    config = C3Config(method="bm25")
    index = E4Index(chunks, config)
    try:
        ranked = index.ranked("MAXIMERA 搁板", {"scope": "ikea_cn"}, 5)
        assert ranked and index.config.corpus_version == "e4_c3_v1"
        assert worker_config(config).model_id == "BAAI/bge-small-zh-v1.5"
    finally:
        index.close()


def test_old_e4_materials_and_request_ledger_unchanged():
    baseline = json.loads(
        (ROOT / ".local/e4_c3_artifacts/historical_baseline.json").read_text("utf-8")
    )
    protected = {
        p: h
        for p, h in baseline.items()
        if p.startswith(("data/knowledge/e4_v1/", "data/tasks/e4_v2/", "data/tasks/private/e4_v2/"))
        or p == ".local/e4_artifacts/generation_budget.sqlite"
    }
    assert protected and all(sha(ROOT / p) == h for p, h in protected.items())
