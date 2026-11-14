"""离线核对来源与业务实体；M0 只读原始快照，不创建业务数据库。"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from aftersalesagent.schemas.retail import RetailSnapshot, SampleReview


def verify_source(root: Path) -> list[str]:
    manifest = json.loads((root / "data/manifests/source_manifest.json").read_text("utf-8"))
    errors: list[str] = []
    for record in manifest["files"]:
        path = (root / record["local_path"]).resolve()
        if not path.is_relative_to(root.resolve()):
            errors.append(f"source path outside project: {record['local_path']}")
            continue
        if not path.is_file():
            errors.append(f"missing source: {record['local_path']}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            errors.append(f"source hash mismatch: {record['local_path']}")
    return errors


def load_snapshot(root: Path) -> RetailSnapshot:
    source = root / "data/upstream/tau2-bench/data/tau2/domains/retail/db.json"
    return RetailSnapshot.model_validate_json(source.read_bytes())


def audit_relations(snapshot: RetailSnapshot) -> list[str]:
    errors: list[str] = []
    for uid, user in snapshot.users.items():
        if uid != user.user_id:
            errors.append(f"user key mismatch: {uid}")
        for oid in user.orders:
            order = snapshot.orders.get(oid)
            if order is None or order.user_id != uid:
                errors.append(f"user-order relation mismatch: {uid}/{oid}")
        for pid, method in user.payment_methods.items():
            if pid != method.id:
                errors.append(f"payment key mismatch: {uid}/{pid}")
    for pid, product in snapshot.products.items():
        if pid != product.product_id:
            errors.append(f"product key mismatch: {pid}")
        for iid, variant in product.variants.items():
            if iid != variant.item_id:
                errors.append(f"variant key mismatch: {pid}/{iid}")
    for oid, order in snapshot.orders.items():
        if oid != order.order_id:
            errors.append(f"order key mismatch: {oid}")
        user = snapshot.users.get(order.user_id)
        if user is None:
            errors.append(f"unknown owner: {oid}")
            continue
        if oid not in user.orders:
            errors.append(f"order absent from owner order list: {oid}")
        for item in order.items:
            product = snapshot.products.get(item.product_id)
            if product is None or item.item_id not in product.variants:
                errors.append(f"unknown order variant: {oid}/{item.item_id}")
        for payment in order.payment_history:
            if payment.payment_method_id not in user.payment_methods:
                errors.append(f"unknown order payment method: {oid}")
    return errors


def audit_samples(root: Path, snapshot: RetailSnapshot) -> list[str]:
    path = root / "data/tasks/m0_dev_reviews.jsonl"
    rows = [SampleReview.model_validate_json(line) for line in path.read_bytes().splitlines()]
    source_root = root / "data/upstream/tau2-bench/data/tau2/domains/retail"
    splits = json.loads((source_root / "split_tasks.json").read_text("utf-8"))
    originals = {
        task["id"]: task for task in json.loads((source_root / "tasks.json").read_text("utf-8"))
    }
    manifest = json.loads((root / "data/manifests/source_manifest.json").read_text("utf-8"))
    errors: list[str] = []
    if len(rows) != 10 or len({r.source_task_id for r in rows}) != 10:
        errors.append("M0 requires ten distinct reviewed development samples")
    for row in rows:
        if row.source_task_id not in splits["train"]:
            errors.append(f"sample outside upstream train: {row.source_task_id}")
        if row.source_commit != manifest["commit"]:
            errors.append(f"sample commit mismatch: {row.sample_id}")
        fixture_path = (root / row.private_fixture).resolve()
        if not fixture_path.is_relative_to((root / "data/tasks/private").resolve()):
            errors.append(f"private fixture path outside private directory: {row.sample_id}")
            continue
        private = json.loads(fixture_path.read_text("utf-8"))
        original = originals[row.source_task_id]
        if private["evaluation_criteria"] != original["evaluation_criteria"]:
            errors.append(f"original criteria changed: {row.sample_id}")
        if private["user_simulator"] != original["user_scenario"]:
            errors.append(f"original user scenario changed: {row.sample_id}")
        if private["initial_state"] != original["initial_state"]:
            errors.append(f"original initial state changed: {row.sample_id}")
        for oid in row.relevant_order_ids:
            order = snapshot.orders.get(oid)
            if order is None or order.user_id != row.user_id:
                errors.append(f"sample owner mismatch: {row.sample_id}/{oid}")
        # 这里只验证参考写动作的静态前提，不把它当作已经运行成功的任务。
        for action in private["evaluation_criteria"].get("actions") or []:
            name, args = action["name"], action["arguments"]
            if name not in {
                "cancel_pending_order",
                "modify_pending_order_address",
                "return_delivered_order_items",
            }:
                continue
            order = snapshot.orders.get(args["order_id"])
            if order is None or order.user_id != row.user_id:
                errors.append(f"reference write owner mismatch: {row.sample_id}")
                continue
            required = "delivered" if name == "return_delivered_order_items" else "pending"
            if order.status != required:
                errors.append(f"reference write status mismatch: {row.sample_id}")
            if name == "cancel_pending_order":
                if args["reason"] not in {"no longer needed", "ordered by mistake"}:
                    errors.append(f"reference cancel reason invalid: {row.sample_id}")
            if name == "return_delivered_order_items":
                available = Counter(item.item_id for item in order.items)
                requested = Counter(args["item_ids"])
                if requested - available:
                    errors.append(f"reference return items invalid: {row.sample_id}")
                method_id = args["payment_method_id"]
                method = snapshot.users[row.user_id].payment_methods.get(method_id)
                original_methods = {p.payment_method_id for p in order.payment_history}
                if method is None or (
                    method_id not in original_methods and method.source != "gift_card"
                ):
                    errors.append(f"reference refund method invalid: {row.sample_id}")
    return errors


def build_report(root: Path) -> dict[str, object]:
    source_errors = verify_source(root)
    # 错误快照不能继续当作有效数据加载和使用。
    if source_errors:
        return {"milestone": "M0", "passed": False, "errors": source_errors}
    snapshot = load_snapshot(root)
    errors = audit_relations(snapshot) + audit_samples(root, snapshot)
    return {
        "milestone": "M0",
        "passed": not errors,
        "source_files_verified": 14,
        "counts": {
            "users": len(snapshot.users),
            "products": len(snapshot.products),
            "orders": len(snapshot.orders),
            "variants": sum(len(p.variants) for p in snapshot.products.values()),
        },
        "order_statuses": dict(Counter(o.status for o in snapshot.orders.values())),
        "duplicate_variant_order_lines": sum(
            len(o.items) - len({i.item_id for i in o.items}) for o in snapshot.orders.values()
        ),
        "reviewed_development_samples": 10,
        "evaluation_mode": "static_source_and_contract_audit",
        "agent_task_success_rate": None,
        "errors": errors,
    }
