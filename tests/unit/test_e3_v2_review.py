"""复核表盲化、哈希绑定和未决项保留，不给真实模型输出补造人工标签。"""

import json
from pathlib import Path

import pytest

from aftersalesagent.evaluation.e3_v2.language import classify
from aftersalesagent.evaluation.e3_v2.review import export_review, fingerprint, import_review
from aftersalesagent.evaluation.e3_v2.scoring import aggregate
from aftersalesagent.evaluation.e3_v2.suite import load_protocol

ROOT = Path(__file__).resolve().parents[2]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def package(tmp_path):
    protocol = load_protocol(ROOT)
    run = tmp_path / "run"
    rows = []
    for group, text in (
        ("S0", "Please provide the new postal code."),
        ("S1", "请提供新的邮政编码。"),
    ):
        record = classify(text, "zh", set(), protocol["language"]["heuristic"])
        record.update(sequence=7, user_context="请用中文帮我处理。")
        row = {
            "task_id": "fixture-task",
            "group": group,
            "protocol_id": protocol["protocol_id"],
            "protocol_version": protocol["version"],
            "protocol_fingerprint": fingerprint(protocol),
            "scoring": {
                "task_status": "success",
                "business": {"success": True},
                "safety": {"passed": True},
                "interaction": {"optional_replies": 1},
                "language": {"records": [record]},
            },
            "usage": {"model_calls": 1, "reported_tokens": 20, "unknown_usage_calls": 0},
        }
        folder = run / group
        folder.mkdir(parents=True)
        save(folder / "result.json", row)
        rows.append(row)
    output = tmp_path / "review"
    export_review(run, output, protocol)
    return protocol, rows, output


def annotate(output, *, uncertain=False):
    sheet_path = output / "review_sheet.json"
    sheet = json.loads(sheet_path.read_text("utf-8"))
    for item in sheet["items"]:
        label = "en" if item["text"].startswith("Please") else "zh"
        item["annotation"] = {
            "label": "undetermined" if uncertain else label,
            "assessment": "needs_review" if uncertain else "pass" if label == "zh" else "fail",
            "rationale": "Offline fixture annotation for import-path tests only.",
        }
    save(sheet_path, sheet)
    return sheet


def do_import(output, protocol, target):
    return import_review(
        output / "review_sheet.json",
        output / "_private_map.json",
        target,
        reviewer="offline-test",
        reviewer_kind="test_fixture",
        protocol=protocol,
    )


def test_blind_export_and_explicit_review_does_not_rewrite_source(tmp_path):
    protocol, rows, output = package(tmp_path)
    sheet = json.loads((output / "review_sheet.json").read_text("utf-8"))
    assert len(sheet["items"]) == 2
    assert all(item["annotation"] is None for item in sheet["items"])
    content = json.dumps(sheet)
    for forbidden in ("S0", "S1", "heuristic_label", "Skill", "result.json", "row_fingerprint"):
        assert forbidden not in content
    before = [p.read_bytes() for p in sorted((tmp_path / "run").glob("*/result.json"))]
    annotate(output)
    result = do_import(output, protocol, tmp_path / "adjudication.json")
    assert result["status"] == "complete" and result["reviewer_kind"] == "test_fixture"
    combined = aggregate(rows, reviewed_language=result)
    assert combined["S0"]["reviewed_language_pass_rate"] == 0
    assert combined["S1"]["reviewed_language_pass_rate"] == 1
    assert before == [p.read_bytes() for p in sorted((tmp_path / "run").glob("*/result.json"))]
    with pytest.raises(ValueError, match="overwrite"):
        do_import(output, protocol, tmp_path / "adjudication.json")


def test_unresolved_reviews_are_not_dropped_from_denominator(tmp_path):
    protocol, rows, output = package(tmp_path)
    annotate(output, uncertain=True)
    result = do_import(output, protocol, tmp_path / "adjudication.json")
    assert result["status"] == "needs_review"
    assert all(g["needs_review"] == 1 and g["pass_rate"] is None for g in result["groups"].values())
    assert all(
        g["reviewed_language_pass_rate"] is None
        for g in aggregate(rows, reviewed_language=result).values()
    )


@pytest.mark.parametrize("fault", ["text", "missing", "duplicate", "rubric", "unreviewed"])
def test_modified_incomplete_or_ambiguous_review_is_rejected(tmp_path, fault):
    protocol, rows, output = package(tmp_path)
    sheet = annotate(output)
    if fault == "text":
        sheet["items"][0]["text"] = "changed reply"
    elif fault == "missing":
        sheet["items"].pop()
    elif fault == "duplicate":
        sheet["items"].append(sheet["items"][0])
    elif fault == "rubric":
        sheet["rubric"]["pass"] = "Always pass."
    else:
        sheet["items"][0]["annotation"] = None
    save(output / "review_sheet.json", sheet)
    with pytest.raises(ValueError):
        do_import(output, protocol, tmp_path / "adjudication.json")


def test_stale_source_or_protocol_rejects_review(tmp_path):
    protocol, rows, output = package(tmp_path)
    annotate(output)
    changed = json.loads(json.dumps(protocol))
    changed["language"]["rubric"]["fail"] += " changed"
    with pytest.raises(ValueError, match="changed"):
        do_import(output, changed, tmp_path / "changed_protocol.json")
    path = tmp_path / "run/S0/result.json"
    path.write_text(path.read_text("utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="source results changed"):
        do_import(output, protocol, tmp_path / "changed_source.json")


def test_adjudication_cannot_be_used_for_other_episode_rows(tmp_path):
    protocol, rows, output = package(tmp_path)
    annotate(output)
    result = do_import(output, protocol, tmp_path / "adjudication.json")
    rows[0]["group"] = "C1"
    with pytest.raises(ValueError, match="different episode"):
        aggregate(rows, reviewed_language=result)
