"""Prepare blinded reviews or calculate a frozen stage gate from explicit judgments."""

import argparse
import json
import random
from pathlib import Path

from aftersalesagent.evaluation.e4.execution import sha
from aftersalesagent.evaluation.e4.suite import load
from aftersalesagent.knowledge.corpus import dump, load_corpus

ROOT = Path(__file__).resolve().parents[1]


def prepare(directory: Path):
    if (directory / "blinded_review.jsonl").exists():
        raise ValueError("Blinded review already prepared; do not reshuffle")
    report = json.loads((directory / "report.json").read_text("utf-8"))
    if not report["complete_split"]:
        raise ValueError("Only complete paired runs enter semantic stage review")
    split = report["config"]["split"]
    cases, _ = load(ROOT, split, report["config"]["suite_version"])
    cases = {c["public"]["task_id"]: c for c in cases}
    chunks, _ = load_corpus(ROOT, "e4_v1")
    by_id = {c.chunk_id: c for c in chunks}
    paths = sorted(directory.glob("E4-*.json"))
    random.SystemRandom().shuffle(paths)
    rows, mapping = [], {}
    for n, path in enumerate(paths, 1):
        original = json.loads(path.read_text("utf-8"))
        case = cases[original["task_id"]]
        result = original["result"]
        review_id = f"BLIND-{n:03}"
        gold = case["criteria"]
        source_ids = (
            set.union(*map(set, gold["support_groups"])) if gold["support_groups"] else set()
        )
        rows.append(
            {
                "review_id": review_id,
                "question": case["public"],
                "answer": result["answer"],
                "cited_sources": result["sources"],
                "expected_status": gold["expected_status"],
                "semantic_checklist": gold["semantic_checklist"],
                "gold_source_excerpts": [by_id[cid].model_dump() for cid in sorted(source_ids)],
                "dimensions": [
                    "all concrete claims source-supported",
                    "conditions and exceptions preserved",
                    "no merchant/simulation or date overreach",
                    "appropriate refusal/conflict",
                    "no transaction claim",
                ],
            }
        )
        mapping[review_id] = {"file": path.name, "sha256": sha(path)}
    (directory / "blinded_review.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), "utf-8"
    )
    (directory / "review_mapping.json").write_bytes(dump(mapping))
    print(json.dumps({"review_items": len(rows), "path": str(directory / "blinded_review.jsonl")}))


def gate(directory: Path):
    protocol_path = ROOT / "configs/evaluation/e4_protocol_v1.json"
    protocol = json.loads(protocol_path.read_text("utf-8"))
    report = json.loads((directory / "report.json").read_text("utf-8"))
    if not report["complete_split"] or report["config"]["protocol_sha256"] != sha(protocol_path):
        raise ValueError("Paired run incomplete or protocol mismatch")
    review = json.loads((directory / "semantic_review.json").read_text("utf-8"))
    mapping = json.loads((directory / "review_mapping.json").read_text("utf-8"))
    if set(review) != set(mapping):
        raise ValueError("Every blinded response needs an explicit semantic review")
    grouped = {label: {} for label in ("L0", protocol["selected"])}
    unresolved = 0
    for rid, item in review.items():
        if item["verdict"] not in {"pass", "fail", "needs_review"}:
            raise ValueError("Unknown semantic review verdict")
        if item["reviewer_kind"] != "assistant" or not item["rationale"].strip():
            raise ValueError("Explicit reviewer type and evidence-based rationale required")
        path = directory / mapping[rid]["file"]
        if sha(path) != mapping[rid]["sha256"]:
            raise ValueError("Reviewed response changed")
        row = json.loads(path.read_text("utf-8"))
        checks = row["automatic"]["checks"]
        unresolved += item["verdict"] == "needs_review"
        row["semantic_review"] = item
        row["success"] = item["verdict"] == "pass" and all(
            checks[key]
            for key in (
                "no_technical_error",
                "status_correct",
                "citation_ids_valid",
                "citation_versions_valid",
                "support_group_cited",
            )
        )
        grouped[row["method"]][row["task_id"]] = row
    left, right = grouped["L0"], grouped[protocol["selected"]]
    cases, _ = load(ROOT, report["config"]["split"], protocol["suite_version"])
    if set(left) != set(right) or len(left) != len(cases):
        raise ValueError("Missing paired tasks")
    improved = [key for key in left if not left[key]["success"] and right[key]["success"]]
    regressed = [key for key in left if left[key]["success"] and not right[key]["success"]]
    statistics = {}
    for label, values in grouped.items():
        rows = list(values.values())
        statistics[label] = {
            "tasks": len(rows),
            "successes": sum(row["success"] for row in rows),
            "state_accuracy": sum(row["automatic"]["checks"]["status_correct"] for row in rows)
            / len(rows),
            "valid_citation_rate": sum(
                row["automatic"]["checks"]["citation_ids_valid"]
                and row["automatic"]["checks"]["citation_versions_valid"]
                for row in rows
            )
            / len(rows),
            "semantic_pass_rate": sum(row["semantic_review"]["verdict"] == "pass" for row in rows)
            / len(rows),
            "technical_errors": sum(row["result"]["error"] is not None for row in rows),
        }
    thresholds = protocol["gates"]
    conditions = {
        "no_unresolved_reviews": unresolved <= thresholds["max_unresolved_reviews"],
        "candidate_improvement": len(improved) >= thresholds["min_candidate_improved_tasks"],
        "no_candidate_regression": len(regressed) <= thresholds["max_candidate_regressed_tasks"],
    }
    retrieval_times = sorted(
        row["result"]["timings"]["retrieval_seconds"] for row in right.values()
    )
    retrieval_p95 = retrieval_times[int((len(retrieval_times) - 1) * 0.95)]
    conditions["candidate_retrieval_resource_ceiling"] = (
        retrieval_p95 <= thresholds["max_retrieval_p95_seconds"]
    )
    for label, stats in statistics.items():
        conditions[label + "_technical"] = (
            stats["technical_errors"] <= thresholds["max_technical_errors"]
        )
        for key in ("state_accuracy", "valid_citation_rate", "semantic_pass_rate"):
            conditions[label + "_" + key] = stats[key] >= thresholds["min_" + key]
    output = {
        "passed": all(conditions.values()),
        "conditions": conditions,
        "statistics": statistics,
        "improved": improved,
        "regressed": regressed,
        "unresolved": unresolved,
        "protocol_sha256": sha(protocol_path),
        "run_dir": directory.relative_to(ROOT).as_posix(),
        "review_sha256": sha(directory / "semantic_review.json"),
        "reviewer_kind": "assistant",
        "candidate_retrieval_p95_seconds": retrieval_p95,
        "not_independent_human_gold": True,
    }
    target = ROOT / f".local/e4_artifacts/{report['config']['split']}_gate.json"
    if target.exists():
        raise ValueError("Stage gate already exists; do not overwrite a previous decision")
    target.write_bytes(dump(output))
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "gate"))
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    directory = args.run_dir.resolve()
    if not directory.is_relative_to(ROOT / ".local/e4_runs"):
        raise ValueError("Review directory escapes E4 runs")
    prepare(directory) if args.action == "prepare" else gate(directory)
