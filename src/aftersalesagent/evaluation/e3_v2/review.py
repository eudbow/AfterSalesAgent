"""盲化语言复核：公开待评表与私有映射分开，导入后另存，绝不改原结果。"""

from __future__ import annotations

import hashlib
import json
import secrets
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from aftersalesagent.schemas.contracts import Contract


def fingerprint(value) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def save(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class Annotation(Contract):
    label: Literal["zh", "en", "mixed", "undetermined"]
    assessment: Literal["pass", "fail", "needs_review"]
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def uncertain_cannot_pass(self) -> Annotation:
        if self.label in {"mixed", "undetermined"} and self.assessment == "pass":
            raise ValueError(
                "Mixed/undetermined explanatory language needs clarification or review"
            )
        return self


def export_review(run_directory: Path, output: Path, protocol: dict) -> dict:
    if output.exists():
        raise ValueError("Refuse to overwrite a review package")
    paths = sorted(run_directory.glob("*/result.json"))
    if not paths:
        raise ValueError("No episode results")
    salt = secrets.token_hex(16)
    items, bindings, sources = [], {}, {}
    for path in paths:
        row = json.loads(path.read_text("utf-8"))
        if row["protocol_id"] != protocol["protocol_id"] or row[
            "protocol_fingerprint"
        ] != fingerprint(protocol):
            raise ValueError("Review source protocol mismatch")
        record_fingerprint = fingerprint(row)
        sources[str(path.resolve())] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "row_fingerprint": record_fingerprint,
        }
        for record in row["scoring"]["language"]["records"]:
            review_id = hashlib.sha256(
                f"{salt}:{path.resolve()}:{record['sequence']}".encode()
            ).hexdigest()[:24]
            visible = {
                "review_id": review_id,
                "expected_language": record["expected_language"],
                "user_context": record["user_context"],
                "text": record["text"],
                "text_sha256": record["text_sha256"],
                "allowed_literals": record["allowed_literals"],
            }
            bindings[review_id] = {
                "visible_fingerprint": fingerprint(visible),
                "group": row["group"],
                "row_fingerprint": record_fingerprint,
                "sequence": record["sequence"],
            }
            items.append(visible | {"annotation": None})
    # 哈希ID顺序打散组别；公开表不含运行路径、组别、Skill或启发式评分。
    items.sort(key=lambda item: item["review_id"])
    sheet = {
        "protocol_id": protocol["protocol_id"],
        "protocol_version": protocol["version"],
        "rubric": protocol["language"]["rubric"],
        "items": items,
    }
    private = {
        "protocol_id": protocol["protocol_id"],
        "protocol_fingerprint": fingerprint(protocol),
        "header_fingerprint": fingerprint(
            {key: value for key, value in sheet.items() if key != "items"}
        ),
        "bindings": bindings,
        "source_files": sources,
        "groups": sorted({json.loads(path.read_text("utf-8"))["group"] for path in paths}),
    }
    output.mkdir(parents=True, exist_ok=False)
    save(output / "review_sheet.json", sheet)
    save(output / "_private_map.json", private)
    return {
        "items": len(items),
        "status": "pending_review",
        "sheet": str(output / "review_sheet.json"),
    }


def import_review(
    sheet_path: Path,
    private_path: Path,
    output: Path,
    *,
    reviewer: str,
    reviewer_kind: Literal["human", "assistant", "test_fixture"],
    protocol: dict,
) -> dict:
    if not reviewer.strip() or reviewer_kind not in {"human", "assistant", "test_fixture"}:
        raise ValueError("An explicit reviewer and reviewer kind are required")
    if output.exists():
        raise ValueError("Refuse to overwrite an adjudication")
    sheet = json.loads(sheet_path.read_text("utf-8"))
    private = json.loads(private_path.read_text("utf-8"))
    if private["protocol_fingerprint"] != fingerprint(protocol):
        raise ValueError("Rubric/protocol changed")
    if (
        fingerprint({key: value for key, value in sheet.items() if key != "items"})
        != private["header_fingerprint"]
    ):
        raise ValueError("Review header/rubric mismatch")
    for path, source in private["source_files"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError("Reviewed source results changed")
    ids = [item["review_id"] for item in sheet["items"]]
    if len(ids) != len(set(ids)) or set(ids) != set(private["bindings"]):
        raise ValueError("Missing, duplicate or foreign review IDs")
    groups: dict[str, dict] = {
        group: {"records": 0, "passed": 0, "failed": 0, "needs_review": 0}
        for group in private["groups"]
    }
    decisions = []
    for item in sheet["items"]:
        visible = {key: value for key, value in item.items() if key != "annotation"}
        binding = private["bindings"][item["review_id"]]
        if fingerprint(visible) != binding["visible_fingerprint"]:
            raise ValueError("Review text/expectation/literals changed")
        annotation = Annotation.model_validate(item["annotation"])
        if annotation.assessment == "pass" and annotation.label != item["expected_language"]:
            raise ValueError("A wrong-language label cannot pass")
        counts = groups[binding["group"]]
        counts["records"] += 1
        counts[
            {"pass": "passed", "fail": "failed", "needs_review": "needs_review"}[
                annotation.assessment
            ]
        ] += 1
        decisions.append(
            {
                "review_id": item["review_id"],
                "sequence": binding["sequence"],
                "row_fingerprint": binding["row_fingerprint"],
                **annotation.model_dump(),
            }
        )
    for counts in groups.values():
        counts["pass_rate"] = (
            counts["passed"] / counts["records"]
            if counts["records"] and not counts["needs_review"]
            else None
        )
    result = {
        "protocol_id": protocol["protocol_id"],
        "protocol_fingerprint": fingerprint(protocol),
        "reviewer": reviewer,
        "reviewer_kind": reviewer_kind,
        "status": "needs_review" if any(g["needs_review"] for g in groups.values()) else "complete",
        "row_fingerprints": sorted(s["row_fingerprint"] for s in private["source_files"].values()),
        "groups": groups,
        "decisions": decisions,
        "note": (
            "Review package separation supports blinding; "
            "local file access is not a security boundary."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    save(output, result)
    return result
