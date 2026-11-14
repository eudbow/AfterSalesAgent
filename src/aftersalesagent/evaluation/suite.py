"""本项目适配任务：公开客户输入/脚本化用户动作与私有判分目标分开保存。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from aftersalesagent.business.source_audit import load_snapshot, verify_source
from aftersalesagent.schemas.contracts import Contract, VisibleTaskInput

COMMIT = "fc0055dc4e0a316c3f83133267fbd6faaa770992"
KINDS = ("query", "address", "cancel", "return", "refuse_cancel", "decline")
SPLITS = ("dev", "validation", "test")
PREFIXES = {
    "dev": "Please handle this customer's structured after-sales request.",
    "validation": "Review my order and help with the following request; ask before making changes.",
    "test": "I need assistance with my order. My details and requested action follow.",
}
TOOL = {
    "address": "modify_pending_order_address",
    "cancel": "cancel_pending_order",
    "return": "return_delivered_order_items",
    "refuse_cancel": "cancel_pending_order",
    "decline": "cancel_pending_order",
}


class CaseCriteria(Contract):
    task_id: str
    kind: Literal["query", "address", "cancel", "return", "refuse_cancel", "decline"]
    user_id: str
    order_id: str
    arguments: dict
    source_commit: str = COMMIT


class Case(Contract):
    public: VisibleTaskInput
    criteria: CaseCriteria
    split: str
    family: str
    approve_matching: bool = True

    @property
    def public_intent(self) -> str:
        return public_data(self.public)["intent"]


def public_data(task: VisibleTaskInput) -> dict:
    return json.loads(task.first_user_message.split("\nREQUEST_DATA:\n", 1)[1])


def build_suite(root: Path) -> tuple[dict[str, bytes], dict]:
    if errors := verify_source(root):
        raise ValueError("; ".join(errors))
    snapshot = load_snapshot(root)
    m0 = [
        json.loads(line)
        for line in (root / "data/tasks/m0_dev_reviews.jsonl").read_text("utf-8").splitlines()
        if line
    ]
    used = {entry["user_id"] for entry in m0}
    candidates = sorted(
        snapshot.orders.values(),
        key=lambda o: hashlib.sha256(("m3-v1:" + o.order_id).encode()).hexdigest(),
    )
    inputs, criteria, memberships = [], [], []
    for split in SPLITS:
        for kind in KINDS:
            for index in range(3):
                states = {"delivered"} if kind in {"return", "refuse_cancel"} else {"pending"}
                order = next(o for o in candidates if o.user_id not in used and o.status in states)
                used.add(order.user_id)
                user = snapshot.users[order.user_id]
                args: dict = {"order_id": order.order_id}
                if kind in {"cancel", "refuse_cancel", "decline"}:
                    args["reason"] = "ordered by mistake" if index % 2 else "no longer needed"
                elif kind == "address":
                    args.update(
                        address1=f"{120 + index} Example Avenue",
                        address2="",
                        city="Seattle",
                        state="WA",
                        country="USA",
                        zip="98101",
                    )
                elif kind == "return":
                    args.update(
                        item_ids=[order.items[0].item_id],
                        payment_method_id=order.payment_history[0].payment_method_id,
                    )
                identity = {
                    "intent": kind,
                    "email": user.email,
                    "order_id": order.order_id,
                    "arguments": args,
                }
                # decline的未来用户拒绝动作不作为额外工具权限；模型仍须先提出确认。
                if kind in {"decline", "refuse_cancel"}:
                    identity["intent"] = "cancel"
                task_id = f"m3-{split}-{kind}-{index + 1}"
                inputs.append(
                    {
                        "task_id": task_id,
                        "first_user_message": PREFIXES[split]
                        + "\nREQUEST_DATA:\n"
                        + json.dumps(identity, ensure_ascii=False, sort_keys=True),
                    }
                )
                criteria.append(
                    CaseCriteria(
                        task_id=task_id,
                        kind=kind,
                        user_id=user.user_id,
                        order_id=order.order_id,
                        arguments=args,
                    ).model_dump(mode="json")
                )
                memberships.append(
                    {
                        "task_id": task_id,
                        "split": split,
                        "kind": kind,
                        "family": f"{split}-{kind}-structured-v1",
                        "user_id": user.user_id,
                        "order_id": order.order_id,
                        "user_decision": "decline"
                        if kind == "decline"
                        else "approve_matching_action",
                    }
                )

    def jsonl(rows):
        return (
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
        ).encode()

    files = {
        "data/tasks/m3/inputs.jsonl": jsonl(inputs),
        "data/tasks/private/m3/criteria.jsonl": jsonl(criteria),
        "data/tasks/m3/user_scripts.jsonl": jsonl(memberships),
    }
    manifest = {
        "suite_id": "aftersales-adapted-retail-v1",
        "frozen_on": "2026-10-19",
        "source_commit": COMMIT,
        "task_count": 54,
        "counts_by_split": {split: 18 for split in SPLITS},
        "counts_by_kind": {kind: 9 for kind in KINDS},
        "files": {path: hashlib.sha256(content).hexdigest() for path, content in files.items()},
        "entity_disjoint": True,
        "m0_users_excluded": True,
        "review": (
            "source/parameter checks plus independent rule and driver audit; "
            "not external human review"
        ),
        "limitations": [
            "Project-authored structured-input tasks, not official tau2 task scores",
            "Prompt families and users/orders are disjoint across local splits",
            "Shared simulator/policy and related task semantics limit statistical independence",
            "Final test is frozen; do not tune prompts/skills on final test results",
        ],
    }
    return files, manifest


def freeze_suite(root: Path) -> dict:
    files, manifest = build_suite(root)
    content = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    files["data/manifests/m3_suite.json"] = content
    for relative, payload in files.items():
        path = root / relative
        if path.exists() and path.read_bytes() != payload:
            raise ValueError("Frozen suite differs; create a new version rather than overwriting")
    for relative, payload in files.items():
        path = root / relative
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
    return manifest


def load_cases(root: Path, split: str) -> tuple[list[Case], dict]:
    manifest = json.loads((root / "data/manifests/m3_suite.json").read_text("utf-8"))
    for relative, digest in manifest["files"].items():
        path = (root / relative).resolve()
        if (
            not path.is_relative_to(root.resolve())
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest
        ):
            raise ValueError("Frozen task file hash mismatch")

    def read(relative):
        return [
            json.loads(line) for line in (root / relative).read_text("utf-8").splitlines() if line
        ]

    visible = {
        row["task_id"]: VisibleTaskInput.model_validate(row)
        for row in read("data/tasks/m3/inputs.jsonl")
    }
    private = {
        row["task_id"]: CaseCriteria.model_validate(row)
        for row in read("data/tasks/private/m3/criteria.jsonl")
    }
    members = read("data/tasks/m3/user_scripts.jsonl")
    if (
        len(visible) != manifest["task_count"]
        or len(private) != len(visible)
        or len(members) != len(visible)
    ):
        raise ValueError("Suite count mismatch")
    groups: dict[str, set] = {}
    for entry in members:
        for key in ("user_id", "order_id", "family"):
            groups.setdefault(key + ":" + entry[key], set()).add(entry["split"])
    if any(len(value) != 1 for value in groups.values()):
        raise ValueError("Entity or template family crosses splits")
    return [
        Case(
            public=visible[m["task_id"]],
            criteria=private[m["task_id"]],
            split=m["split"],
            family=m["family"],
            approve_matching=m["user_decision"] != "decline",
        )
        for m in members
        if m["split"] == split
    ], manifest
