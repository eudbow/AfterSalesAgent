"""冻结快照与初始样例的回归检查，不执行上游代码、不联网。"""

import hashlib
import json
from pathlib import Path

from aftersalesagent.business.source_audit import (
    audit_relations,
    audit_samples,
    load_snapshot,
    verify_source,
)
from aftersalesagent.schemas.contracts import VisibleTaskInput

ROOT = Path(__file__).resolve().parents[2]


def test_all_pinned_files_have_original_hashes() -> None:
    assert verify_source(ROOT) == []


def test_source_hash_detects_modified_data(tmp_path: Path) -> None:
    path = tmp_path / "data/source.json"
    path.parent.mkdir()
    path.write_text("original")
    manifest_path = tmp_path / "data/manifests/source_manifest.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "local_path": "data/source.json",
                        "sha256": hashlib.sha256(b"original").hexdigest(),
                    }
                ]
            }
        )
    )
    assert verify_source(tmp_path) == []
    path.write_text("changed")
    assert "hash mismatch" in verify_source(tmp_path)[0]


def test_all_entities_preserve_valid_relationships() -> None:
    snapshot = load_snapshot(ROOT)
    assert audit_relations(snapshot) == []
    assert len(snapshot.users) == 500
    assert len(snapshot.orders) == 1000
    assert len(snapshot.products) == 50


def test_relationship_audit_detects_missing_owner() -> None:
    snapshot = load_snapshot(ROOT).model_copy(deep=True)
    first = next(iter(snapshot.orders.values()))
    snapshot.users.pop(first.user_id)
    assert any("unknown owner" in error for error in audit_relations(snapshot))


def test_duplicate_order_lines_are_not_collapsed() -> None:
    snapshot = load_snapshot(ROOT)
    duplicates = sum(
        len(o.items) - len({i.item_id for i in o.items}) for o in snapshot.orders.values()
    )
    assert duplicates == 7


def test_ten_reviews_remain_in_train_and_match_source() -> None:
    assert audit_samples(ROOT, load_snapshot(ROOT)) == []


def test_public_samples_contain_only_visible_inputs() -> None:
    path = ROOT / "data/tasks/m0_dev_inputs.jsonl"
    rows = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
    assert len(rows) == 10
    for row in rows:
        visible = VisibleTaskInput.model_validate(row)
        assert visible.first_user_message
        assert set(row) == {"task_id", "first_user_message"}


def test_local_tool_specs_match_argument_shapes() -> None:
    import ast

    from aftersalesagent.schemas.contracts import ToolSpec

    source = ROOT / "data/upstream/tau2-bench/src/tau2/domains/retail/tools.py"
    methods = {
        node.name: node
        for node in ast.walk(ast.parse(source.read_text("utf-8")))
        if isinstance(node, ast.FunctionDef)
    }
    rows = json.loads((ROOT / "data/manifests/tool_contracts.json").read_text("utf-8"))
    for row in rows:
        spec = ToolSpec.model_validate(row["spec"])
        method = methods[spec.name]
        expected = {a.arg for a in method.args.args if a.arg != "self"}
        properties = spec.arguments_schema["properties"]
        required = spec.arguments_schema["required"]
        assert isinstance(properties, dict)
        assert isinstance(required, list)
        assert set(properties) == expected
        assert set(required) == expected
