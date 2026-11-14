"""离线核对E1来源、36项任务、中文检索开发基准；不加载.env、不调用模型。"""

import hashlib
import json
import sqlite3
from pathlib import Path

from aftersalesagent.business.source_audit import verify_source
from aftersalesagent.evaluation.e1_suite import load
from aftersalesagent.knowledge.corpus import eligible, load_corpus
from aftersalesagent.knowledge.retrieval import SQLiteKnowledgeIndex

ROOT = Path(__file__).resolve().parents[1]


def main():
    chunks, manifest = load_corpus(ROOT)
    audit = []
    for split in ("dev", "validation", "test"):
        cases, _ = load(ROOT, split)
        for case in cases:
            public, criteria = case["public"], case["criteria"]
            allowed = eligible(chunks, public["filters"])
            by_id = {c.chunk_id: c for c in allowed}
            assert set(criteria["required_ids"]) <= set(by_id)
            versions = {by_id[c].version for c in criteria["required_ids"]}
            assert versions == set(criteria["versions"])
            audit.append(public["task_id"])
    benchmarks = {}
    cases, _ = load(ROOT, "dev")
    for tokenizer in ("unicode61", "trigram", "char"):
        index = SQLiteKnowledgeIndex(chunks, tokenizer)
        rows = []
        for case in cases:
            public, criteria = case["public"], case["criteria"]
            gold = set(criteria["required_ids"])
            found = index.retrieve(public["question"], public["filters"], 5)
            ids = [c.chunk_id for c in found]
            rows.append(
                {
                    "task_id": public["task_id"],
                    "required_ids": sorted(gold),
                    "retrieved_ids": ids,
                    "recall": len(gold & set(ids)) / len(gold) if gold else None,
                }
            )
        recalls = [r["recall"] for r in rows if r["recall"] is not None]
        benchmarks[tokenizer] = {
            "macro_recall_at_5": sum(recalls) / len(recalls),
            "denominator": len(recalls),
            "rows": rows,
        }
        index.close()
    protected = json.loads((ROOT / ".local/results/e1_protected_files.json").read_text("utf-8"))
    unchanged = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
        for name, expected in protected.items()
    }
    source_errors = verify_source(ROOT)
    assert all(unchanged.values()), unchanged
    assert not source_errors, source_errors
    report = {
        "passed": True,
        "sqlite_version": sqlite3.sqlite_version,
        "chunks": manifest["chunk_count"],
        "tasks_reviewed": len(audit),
        "source_checks": "all manifest hashes and source span hashes verified",
        "protected_files_unchanged": unchanged,
        "original_retail_source_errors": source_errors,
        "tokenizer_benchmarks_dev_only": benchmarks,
        "model_calls": 0,
        "gpu_experiments": 0,
    }
    path = ROOT / ".local/results/e1_offline_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", "utf-8")
    print(
        json.dumps(
            {
                "passed": True,
                "chunks": len(chunks),
                "tasks": len(audit),
                "dev_recall_at_5": {k: v["macro_recall_at_5"] for k, v in benchmarks.items()},
                "protected_files_unchanged": len(unchanged),
                "report": str(path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
