"""独立自然语言任务v1：首条消息、用户计划、私有目标分离；不改M3任务。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from aftersalesagent.business.source_audit import load_snapshot, verify_source
from aftersalesagent.evaluation.suite import COMMIT, CaseCriteria
from aftersalesagent.schemas.contracts import Contract, VisibleTaskInput

SPLITS = ("dev", "validation", "test")
CHALLENGES = ("complete", "format", "missing", "correction", "confirmation", "refusal")


class UserPlan(Contract):
    task_id: str
    split: Literal["dev", "validation", "test"]
    challenge: str
    family: str
    source_user_id: str
    order_id: str
    tool_name: str | None
    initial_arguments: dict
    final_arguments: dict
    missing_slot: Literal["reason", "zip"] | None = None
    answer: str | None = None
    correction: str | None = None
    plain_yes: str | None = None
    approve: bool = True
    max_messages: int = 4


class NLCase(Contract):
    public: VisibleTaskInput
    plan: UserPlan
    criteria: CaseCriteria


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def build(root: Path) -> tuple[dict[str, bytes], dict]:
    if errors := verify_source(root):
        raise ValueError("; ".join(errors))
    snapshot = load_snapshot(root)
    used = {
        json.loads(line)["user_id"]
        for line in (root / "data/tasks/m0_dev_reviews.jsonl").read_text("utf-8").splitlines()
    }
    used |= {
        json.loads(line)["user_id"]
        for line in (root / "data/tasks/m3/user_scripts.jsonl").read_text("utf-8").splitlines()
    }
    excluded = sorted(used)
    orders = sorted(
        snapshot.orders.values(),
        key=lambda o: hashlib.sha256(("nl-v1:" + o.order_id).encode()).hexdigest(),
    )
    inputs, plans, targets = [], [], []
    for split_index, split in enumerate(SPLITS):
        intro = (
            "你好，麻烦帮我处理一下。",
            "想请你协助处理一笔售后。",
            "这边有个订单需要你帮忙看看。",
        )[split_index]
        for index in range(12):
            challenge = CHALLENGES[index // 2]
            kind = (
                "query",
                "return",
                "query",
                "cancel",
                "cancel",
                "address",
                "address",
                "address",
                "decline",
                "return",
                "refuse_cancel",
                "refuse_cancel",
            )[index]
            state = "delivered" if kind in {"return", "refuse_cancel"} else "pending"
            order = next(o for o in orders if o.user_id not in used and o.status == state)
            used.add(order.user_id)
            user = snapshot.users[order.user_id]
            oid = order.order_id
            args: dict = {"order_id": oid}
            if kind in {"cancel", "decline", "refuse_cancel"}:
                args["reason"] = "ordered by mistake"
            if kind == "address":
                args.update(
                    address1=f"{510 + split_index * 20 + index} Cedar Street",
                    address2="",
                    city="Seattle",
                    state="WA",
                    country="USA",
                    zip="98101",
                )
            if kind == "return":
                args.update(
                    item_ids=[order.items[0].item_id],
                    payment_method_id=order.payment_history[0].payment_method_id,
                )
            initial_args = dict(args)
            slot, answer, correction, plain_yes = None, None, None, None
            identity = f"我的邮箱是 {user.email}。"
            if index == 0:
                request = f"请帮我查一下订单 {oid} 现在是什么状态。"
            elif index == 1:
                request = (
                    f"想退订单 {oid} 里的商品，商品编号 {args['item_ids'][0]}，"
                    f"退回原付款方式 {args['payment_method_id']}。"
                )
            elif index == 2:
                request = f"订单号是「 {oid} 」，两侧空格不是编号，请查一下现在到哪一步了。"
            elif index == 3:
                request = f"麻烦把订单（{oid}）取消，是我下错单了。编号里的#请保留。"
            elif index == 4:
                request = f"我想取消订单 {oid}。"
                slot, answer = "reason", "取消原因是我下错单了。"
            elif index == 5:
                request = (
                    f"请改订单 {oid} 的收货地址：{args['address1']}，"
                    "第二行留空，Seattle，WA，USA。邮编我还没告诉你，请先问我，不要猜。"
                )
                slot, answer = "zip", "新地址的邮编是98101。"
            elif index in {6, 7}:
                request = (
                    f"请把订单 {oid} 的地址改成 {args['address1']}，"
                    "第二行留空，Seattle，WA，USA，邮编98101。"
                )
                if index == 6:
                    args["zip"] = "98109"
                    correction = "先别执行，我把新地址的邮编改为98109，其余地址字段不变。"
                else:
                    args.update(city="Tacoma", zip="98402")
                    correction = (
                        "还没确认，刚才的新地址应改成Tacoma，邮编98402，"
                        "街道、州、国家等其余字段不变。"
                    )
            elif index == 8:
                request = f"订单 {oid} 是我误买的，帮我发起取消。"
            elif index == 9:
                request = (
                    f"我要退订单 {oid} 的商品 {args['item_ids'][0]}，"
                    f"退款选原付款方式 {args['payment_method_id']}。"
                )
                plain_yes = "好的，就按刚才这个方案办。"
            elif index == 10:
                request = f"不想要订单 {oid} 了，是我误买的，能帮我取消吗？"
            else:
                request = f"订单 {oid} 我下错了，即使已经送到也想直接取消，请按规则帮我看看。"
            task_id = f"nl1-{split}-{index + 1:02d}"
            tool = {
                "cancel": "cancel_pending_order",
                "decline": "cancel_pending_order",
                "refuse_cancel": "cancel_pending_order",
                "address": "modify_pending_order_address",
                "return": "return_delivered_order_items",
            }.get(kind)
            inputs.append({"task_id": task_id, "first_user_message": intro + identity + request})
            plans.append(
                UserPlan(
                    task_id=task_id,
                    split=split,
                    challenge=challenge,
                    family=f"{split}-{challenge}-wording-v1",
                    source_user_id=user.user_id,
                    order_id=oid,
                    tool_name=tool,
                    initial_arguments=initial_args,
                    final_arguments=args,
                    missing_slot=slot,
                    answer=answer,
                    correction=correction,
                    plain_yes=plain_yes,
                    approve=kind != "decline",
                ).model_dump(mode="json")
            )
            targets.append(
                CaseCriteria(
                    task_id=task_id,
                    kind=kind,
                    user_id=user.user_id,
                    order_id=oid,
                    arguments=args,
                ).model_dump(mode="json")
            )

    def lines(rows):
        return "".join(
            json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows
        ).encode()

    files = {
        "data/tasks/nl_v1/inputs.jsonl": lines(inputs),
        "data/tasks/nl_v1/user_scripts.jsonl": lines(plans),
        "data/tasks/private/nl_v1/criteria.jsonl": lines(targets),
    }
    manifest = {
        "suite_id": "aftersales-natural-language-v1",
        "frozen_on": "2026-10-22",
        "source_commit": COMMIT,
        "task_count": 36,
        "counts_by_split": {s: 12 for s in SPLITS},
        "counts_by_challenge": {s: 6 for s in CHALLENGES},
        "excluded_m0_m3_users": excluded,
        "files": {p: hashlib.sha256(b).hexdigest() for p, b in files.items()},
        "review": "Programmatic provenance and assistant review, not independent human annotation",
        "limitations": [
            "Project-authored Chinese tasks and bounded scripted user, "
            "not real customer distribution",
            "Split-specific wording but shared challenge semantics; not statistically independent",
            "36 tasks are a pilot, not an official tau2 score",
            "Missing-slot question and final-reply evidence use coarse bilingual rules",
            "No paired structured control in this first version",
        ],
    }
    return files, manifest


def freeze(root: Path) -> dict:
    files, manifest = build(root)
    files["data/manifests/nl_v1_suite.json"] = json_bytes(manifest)
    for rel, content in files.items():
        p = root / rel
        if p.exists() and p.read_bytes() != content:
            raise ValueError("Frozen NL suite differs; create another version")
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            p.write_bytes(content)
    return manifest


def load(root: Path, split: str) -> tuple[list[NLCase], dict]:
    manifest = json.loads((root / "data/manifests/nl_v1_suite.json").read_text("utf-8"))
    rows = {}
    for rel, sha in manifest["files"].items():
        p = (root / rel).resolve()
        if (
            not p.is_relative_to(root.resolve())
            or hashlib.sha256(p.read_bytes()).hexdigest() != sha
        ):
            raise ValueError("NL file hash mismatch")
        items = [json.loads(line) for line in p.read_text("utf-8").splitlines()]
        rows[p.name if p.name != "inputs.jsonl" else "inputs"] = {r["task_id"]: r for r in items}
    inputs, scripts, criteria = rows["inputs"], rows["user_scripts.jsonl"], rows["criteria.jsonl"]
    if not (len(inputs) == len(scripts) == len(criteria) == 36) or not (
        inputs.keys() == scripts.keys() == criteria.keys()
    ):
        raise ValueError("NL suite counts/IDs mismatch")
    groups = {}
    for plan in scripts.values():
        if plan["source_user_id"] in manifest["excluded_m0_m3_users"]:
            raise ValueError("NL user leaked from previous suite")
        for field in ("source_user_id", "order_id", "family"):
            groups.setdefault(field + ":" + plan[field], set()).add(plan["split"])
    if any(len(group) != 1 for group in groups.values()):
        raise ValueError("NL split leakage")
    cases = [
        NLCase(
            public=VisibleTaskInput.model_validate(inputs[k]),
            plan=UserPlan.model_validate(scripts[k]),
            criteria=CaseCriteria.model_validate(criteria[k]),
        )
        for k in inputs
        if scripts[k]["split"] == split
    ]
    if len(cases) != 12:
        raise ValueError("NL split count mismatch")
    return cases, manifest
