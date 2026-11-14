"""Load public inputs and private E4 acceptance separately; verify manifest and split membership."""

import hashlib
import json
import re
from pathlib import Path


def load(root: Path, split: str, version: str = "e4_v2"):
    if split not in {"dev", "validation", "test"}:
        raise ValueError("Unknown E4 split")
    if version not in {"e4_v1", "e4_v2"}:
        raise ValueError("Unknown E4 suite version")
    manifest_path = root / f"data/manifests/e4_suite_{version.removeprefix('e4_')}.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    for relative, expected in manifest["files"].items():
        path = (root / relative).resolve()
        if (
            not path.is_relative_to(root.resolve())
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise ValueError("E4 material hash mismatch")
    public = [
        json.loads(line)
        for line in (root / f"data/tasks/{version}/{split}.jsonl").read_text("utf-8").splitlines()
    ]
    private = [
        json.loads(line)
        for line in (root / f"data/tasks/private/{version}/{split}.jsonl")
        .read_text("utf-8")
        .splitlines()
    ]
    by_id = {row["task_id"]: row for row in private}
    if len(by_id) != len(private) or {row["task_id"] for row in public} != by_id.keys():
        raise ValueError("E4 public/private task mismatch")
    if len({row["task_id"] for row in public}) != len(public):
        raise ValueError("Duplicate E4 public task ID")
    cases = []
    for row in public:
        criteria = dict(by_id[row["task_id"]])
        tags = ["official_scope" if row["filters"]["scope"] == "ikea_cn" else "project_simulation"]
        if row["filters"].get("as_of"):
            tags.append("explicit_date_filter")
        if criteria["expected_status"] == "insufficient_evidence":
            tags.append("answerability_negative")
        if criteria["expected_status"] == "conflict":
            tags.append("conflicting_sources")
        if criteria["support_groups"] and min(map(len, criteria["support_groups"])) > 1:
            tags.append("multiple_required_chunks")
        if re.search(r"[A-Z]{3,}", row["question"]):
            tags.append("latin_identifier")
        criteria["secondary_tags"] = tags
        cases.append({"public": row, "criteria": criteria})
    return cases, manifest
