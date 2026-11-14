"""Freeze the retrieval challenger and gates before real paired answers."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from aftersalesagent.evaluation.e4.execution import sha, source_fingerprints, verify_fingerprints
from aftersalesagent.knowledge.corpus import dump
from aftersalesagent.local_config import load_local_config, options_from_settings

ROOT = Path(__file__).resolve().parents[1]


def main(args):
    args.retrieval_report = args.retrieval_report.resolve()
    target = ROOT / "configs/evaluation/e4_protocol_v1.json"
    if target.exists():
        raise ValueError("E4 protocol already frozen; no overwrite")
    report = json.loads(args.retrieval_report.read_text("utf-8"))
    if set(report["methods"]) != {"L0", "D0", "H0", "H1"}:
        raise ValueError("Complete development retrieval matrix required")
    candidates = [
        label for label in ("D0", "H0", "H1") if report["methods"][label]["p95_seconds"] <= 30
    ]
    if not candidates:
        raise ValueError("No enhanced candidate meets the local resource ceiling")
    selected = max(
        candidates,
        key=lambda label: (
            report["methods"][label]["metrics"]["complete_support"],
            report["methods"][label]["metrics"]["ndcg_at_5"],
            -report["methods"][label]["p95_seconds"],
        ),
    )
    files = source_fingerprints(ROOT)
    for model in ("bge-small-zh-v1.5", "bge-reranker-v2-m3"):
        lock_path = ROOT / f".local/e4_artifacts/{model}.json"
        lock = json.loads(lock_path.read_text("utf-8"))
        for item in lock["files"]:
            path = ROOT / lock["local_path"] / item["name"]
            if sha(path) != item["sha256"]:
                raise ValueError("Model file changed after download")
            files[path.relative_to(ROOT).as_posix()] = item["sha256"]
        files[lock_path.relative_to(ROOT).as_posix()] = sha(lock_path)
    for path in [
        *ROOT.joinpath("configs/evaluation").glob("e4_*_v1.json"),
        ROOT / ".local/e4_artifacts/vectors.json",
        ROOT / "data/manifests/e4_suite_v2.json",
    ]:
        files[path.relative_to(ROOT).as_posix()] = sha(path)
    settings, _ = load_local_config(ROOT / ".env")
    options = options_from_settings(settings).model_copy(update={"timeout_seconds": 60.0})
    protocol = {
        "version": "e4_protocol_v1",
        "suite_version": "e4_v2",
        "evidence_fields_counted": ["section", "text"],
        "created_at": datetime.now(UTC).isoformat(),
        "selected": selected,
        "methods": {"L0": "bm25", "D0": "dense", "H0": "hybrid", "H1": "rerank"},
        "selection_report": args.retrieval_report.relative_to(ROOT).as_posix(),
        "selection_report_sha256": sha(args.retrieval_report),
        "options": options.model_dump(),
        "files": files,
        "request_limit": 300,
        "max_output_tokens": options.max_output_tokens,
        "top_k": 5,
        "context_char_limit": 50000,
        "skill": "none",
        "transport": "native",
        "smoke_task_ids": ["E4-D01-1", "E4-D05-1", "E4-D07-1", "E4-D11-1", "E4-D19-1", "E4-D20-1"],
        "gates": {
            "max_technical_errors": 0,
            "min_state_accuracy": 0.90,
            "min_valid_citation_rate": 1.0,
            "min_semantic_pass_rate": 0.85,
            "min_candidate_improved_tasks": 1,
            "max_candidate_regressed_tasks": 0,
            "max_retrieval_p95_seconds": 30.0,
            "max_unresolved_reviews": 0,
        },
        "validation_rule": (
            "Development gate and complete blinded assistant review before validation; "
            "validation gate before test"
        ),
        "default_change_rule": (
            "Keep BM25 unless paired quality improves with zero applicability/provenance "
            "errors and resource ceiling passes"
        ),
        "price_estimate": None,
        "price_note": "Record actual usage; no invoice-equivalent currency total claimed",
    }
    verify_fingerprints(ROOT, files)
    target.write_bytes(dump(protocol))
    print(json.dumps({"protocol": str(target), "selected": selected, "request_limit": 300}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-report", type=Path, required=True)
    main(parser.parse_args())
