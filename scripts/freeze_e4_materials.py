"""Seal authored E4 material before retrieval tuning, preserving existing source and task hashes."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from aftersalesagent.knowledge.corpus import dump, load_corpus

ROOT = Path(__file__).resolve().parents[1]


def main():
    target = ROOT / "data/manifests/e4_suite_v1.json"
    if target.exists():
        raise SystemExit("E4 suite already sealed; never overwrite")
    chunks, corpus = load_corpus(ROOT, "e4_v1")
    valid_ids = {chunk.chunk_id for chunk in chunks}
    report = json.loads((ROOT / "data/tasks/private/e4_v1/review.json").read_text("utf-8"))
    if report["cross_split_near_duplicates"]:
        raise ValueError("Near-duplicate review remains unresolved")
    owners = {}
    for split in ("dev", "validation", "test"):
        path = ROOT / f"data/tasks/private/e4_v1/{split}.jsonl"
        for row in map(json.loads, path.read_text("utf-8").splitlines()):
            owner = owners.setdefault(row["family_id"], split)
            if owner != split or row["review_status"] != "source_checked":
                raise ValueError("Family leakage or unresolved review")
            if any(not set(group) <= valid_ids for group in row["support_groups"]):
                raise ValueError("Unknown supporting evidence")
            if row["expected_status"] != "insufficient_evidence" and not row["support_groups"]:
                raise ValueError("Missing supporting evidence")
    folders = [
        ROOT / "data/tasks/e4_v1",
        ROOT / "data/tasks/private/e4_v1",
        ROOT / "data/knowledge/e4_v1",
    ]
    files = {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for folder in folders
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }
    target.parent.mkdir(exist_ok=True, parents=True)
    target.write_bytes(
        dump(
            {
                "suite_version": "e4_v1",
                "created_at": datetime.now(UTC).isoformat(),
                "files": files,
                "splits": report["splits"],
                "fact_families": 50,
                "reviewer_kind": "assistant",
                "independent_human_gold": False,
                "challenge_counts": report["challenge_counts"],
                "limitations": [
                    "Three within-family phrasings, not 150 independent facts",
                    "Warranty-heavy mainland-China self-built acceptance",
                    "Not a production or official benchmark",
                    "Shared domain primitives remain across splits",
                ],
                "corpus_chunks": corpus["chunks"],
            }
        )
    )
    print(
        json.dumps(
            {
                "material_files": len(files),
                "tasks": sum(report["splits"].values()),
                "families": len(owners),
            }
        )
    )


if __name__ == "__main__":
    main()
