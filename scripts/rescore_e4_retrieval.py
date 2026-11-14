"""Re-score unchanged real rankings against revised gold; no new inference or API calls."""

import argparse
import json
import shutil
from pathlib import Path

from aftersalesagent.evaluation.e4.execution import sha, source_fingerprints
from aftersalesagent.evaluation.e4.scoring import retrieval_metrics, summarize
from aftersalesagent.evaluation.e4.suite import load
from aftersalesagent.knowledge.corpus import dump

ROOT = Path(__file__).resolve().parents[1]


def main(directory, output_tag):
    directory = directory.resolve()
    if not directory.is_relative_to(ROOT / ".local/e4_runs"):
        raise ValueError("Run directory escapes project")
    original = json.loads((directory / "report.json").read_text("utf-8"))
    if set(original["methods"]) != {"L0", "D0", "H0", "H1"}:
        raise ValueError("Complete real-model matrix required before re-scoring")
    for split in ("dev", "validation", "test"):
        old = ROOT / f"data/tasks/e4_v1/{split}.jsonl"
        new = ROOT / f"data/tasks/e4_v2/{split}.jsonl"
        if old.read_bytes() != new.read_bytes():
            raise ValueError("Re-scoring requires byte-identical public model inputs")
    cases, manifest = load(ROOT, original["split"], "e4_v2")
    gold = {c["public"]["task_id"]: c["criteria"] for c in cases}
    if output_tag not in {"draft", "final"}:
        raise ValueError("Unknown annotation output tag")
    target = directory / ("gold_e4_v2_" + output_tag)
    target.mkdir(exist_ok=False)
    result = {
        **original,
        "suite_manifest": manifest,
        "annotation_update": {
            "gold_version": "e4_v2",
            "questions_changed": 0,
            "corpus_changed": False,
            "new_model_calls": 0,
            "raw_report_sha256": sha(directory / "report.json"),
            "raw_run": directory.relative_to(ROOT).as_posix(),
            "rescoring_source_hashes": source_fingerprints(ROOT),
        },
        "methods": {},
    }
    for label, previous in original["methods"].items():
        rows = [
            json.loads(line)
            for line in (directory / (label + ".jsonl")).read_text("utf-8").splitlines()
        ]
        if len(rows) != len(cases) or {r["task_id"] for r in rows} != gold.keys():
            raise ValueError("Raw ranking matrix is incomplete")
        for row in rows:
            row["metrics"] = retrieval_metrics(row["ids"], gold[row["task_id"]])
            row["secondary_tags"] = gold[row["task_id"]]["secondary_tags"]
        (target / (label + ".jsonl")).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), "utf-8"
        )
        challenges = {
            tag: summarize([row for row in rows if row["challenge"] == tag])
            for tag in sorted({row["challenge"] for row in rows})
        }
        result["methods"][label] = {
            **previous,
            "metrics": summarize(rows),
            "challenges": challenges,
        }
    (target / "report.json").write_bytes(dump(result))
    for relative in result["annotation_update"]["rescoring_source_hashes"]:
        path = target / "snapshot" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, path)
    print(
        json.dumps(
            {
                "report": str(target / "report.json"),
                "metrics": {key: value["metrics"] for key, value in result["methods"].items()},
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-tag", choices=("draft", "final"), default="final")
    args = parser.parse_args()
    main(args.run_dir, args.output_tag)
