"""Retail source integrity and tenant isolation regression checks."""

import importlib.util
from pathlib import Path

import pytest

from aftersalesagent.knowledge.corpus import eligible, load_corpus
from aftersalesagent.knowledge.retrieval import SQLiteKnowledgeIndex

ROOT = Path(__file__).resolve().parents[2]


def test_rebuild_is_byte_identical():
    spec = importlib.util.spec_from_file_location(
        "retail_builder", ROOT / "scripts/prepare_retail_corpus.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    before = {p: p.read_bytes() for p in module.BASE.rglob("*") if p.is_file()}
    module.build()
    assert before == {p: p.read_bytes() for p in before}


def test_tampered_source_is_rejected(tmp_path):
    import json
    import shutil

    base = ROOT / "data/knowledge/retail_sim_v1"
    manifest = json.loads((base / "manifest.json").read_text("utf-8"))
    for rel in [*manifest["files"], "data/knowledge/retail_sim_v1/manifest.json"]:
        dest = tmp_path / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, dest)
    target = tmp_path / "data/knowledge/retail_sim_v1/documents/05_returns.md"
    target.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_corpus(tmp_path, "retail_sim_v1")


def test_mixed_index_filters_before_top_k():
    retail, _ = load_corpus(ROOT, "retail_sim_v1")
    legacy, _ = load_corpus(ROOT, "e1_v1")
    index = SQLiteKnowledgeIndex(retail + legacy)
    try:
        found = index.retrieve("退款保证退货", {"scope": "retail_sim"}, 30)
        assert found and all(c.scope == "retail_sim" for c in found)
        assert not eligible(retail, {"scope": "retail_sim", "as_of": "2026-11-14"})
        with pytest.raises(ValueError):
            eligible(retail, {"scope": "typo"})
        with pytest.raises(ValueError):
            load_corpus(ROOT, "typo")
    finally:
        index.close()
