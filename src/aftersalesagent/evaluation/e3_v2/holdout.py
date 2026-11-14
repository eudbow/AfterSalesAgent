"""E3 v2留出材料：24个新组合，仅离线审核；本轮真实基线只使用开发集。"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Literal

from aftersalesagent.business.service import ARGUMENTS
from aftersalesagent.business.source_audit import load_snapshot
from aftersalesagent.evaluation.e3_v2.contracts import (
    SLOT_FIELD,
    FactUpdate,
    Intent,
    Stage,
    UserScript,
    apply_update,
)
from aftersalesagent.evaluation.e3_v2.review import fingerprint
from aftersalesagent.evaluation.e3_v2.suite import load as load_dev
from aftersalesagent.evaluation.e3_v2.suite import sha
from aftersalesagent.evaluation.suite import CaseCriteria
from aftersalesagent.schemas.contracts import VisibleTaskInput

CANCEL = "cancel_pending_order"
ADDRESS = "modify_pending_order_address"
RETURN = "return_delivered_order_items"
NAMES = {
    "validation": [
        "退货改取消被拒后重建退货",
        "改址转取消补原因再更正",
        "取消转缺邮编改址后清空第二行",
        "选退货清单后改礼品卡再缩减",
        "改址文字同意后跨州修正",
        "取消文字同意后改原因",
        "退货文字同意后更换退款方式",
        "补邮编改街道后文字同意",
        "补取消原因后切换完整改址",
        "选择退款方式后替换退货商品",
        "中英切换改址取消再改址",
        "跨语言清单修改文字同意再改支付",
    ],
    "test": [
        "补取消原因转缺邮编改址再联合更正",
        "取消被拒后选退货商品并文字同意",
        "改礼品卡扩展清单再替换全部商品",
        "缺邮编改址转取消再补原因",
        "取消改址再取消并更改原因",
        "选择退款方式缩减商品再更换支付",
        "地址第二行清空恢复后文字同意",
        "取消改原因文字同意后转改址",
        "选退货商品更换礼品卡后文字同意",
        "补邮编整体替换地址后文字同意",
        "跨语言选商品更换支付再替换商品",
        "跨语言补原因更正再转缺邮编改址",
    ],
}


def intent(tool, arguments):
    return Intent.model_validate({"tool_name": tool, "arguments": deepcopy(arguments)})


def item_list(arguments):
    values = arguments.get("item_ids", [])
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError("Expected item ID strings")
    return [str(value) for value in values]


def item_set(arguments):
    return set(item_list(arguments))


def signature(script: UserScript):
    """按实际工具、触发类型和字段变化分组；语言/订单号/措辞不作为隔离证据。"""
    state = script.initial_intent
    stages = []
    for stage in script.stages:
        after = apply_update(state, stage.update)
        before_items = item_set(state.arguments)
        after_items = item_set(after.arguments)
        stages.append(
            {
                "trigger": stage.trigger,
                "slot": stage.slot,
                "tool_before": state.tool_name,
                "tool_after": after.tool_name,
                "changed_fields": sorted(stage.update.arguments),
                "item_set_transition": {
                    "before_known": "item_ids" in state.arguments,
                    "before_count": len(before_items),
                    "after_count": len(after_items),
                    "added": len(after_items - before_items),
                    "removed": len(before_items - after_items),
                }
                if "item_ids" in stage.update.arguments
                else None,
                "cleared_fields": sorted(
                    k for k, value in stage.update.arguments.items() if value == ""
                ),
            }
        )
        state = after
    return fingerprint(
        {
            "initial_tool": script.initial_intent.tool_name,
            "initial_known_fields": sorted(script.initial_intent.arguments),
            "stages": stages,
            "final_tool": script.final_intent.tool_name,
        }
    )


def describe(value: Intent, language):
    a = value.arguments
    oid = a["order_id"]
    if value.tool_name == CANCEL:
        reason = a.get("reason")
        if reason is not None and not isinstance(reason, str):
            raise ValueError("Expected a cancellation reason string")
        zh = {"no longer needed": "我已经不需要了", "ordered by mistake": "是我下错单了"}
        if language == "zh":
            suffix = zh[reason] if reason else "原因尚未决定，请先问我，不要替我选择"
            return f"请为订单{oid}准备取消；{suffix}。"
        suffix = f"the reason is {reason}" if reason else "the reason is undecided; please ask me"
        return f"Prepare cancellation of order {oid}; {suffix}."
    if value.tool_name == ADDRESS:
        line2 = a.get("address2", "")
        if language == "zh":
            postcode = f"邮编{a['zip']}" if "zip" in a else "邮编还未提供，请先询问，不要猜"
            return (
                f"请将订单{oid}地址改为{a['address1']}，第二行{line2 or '留空'}，"
                f"{a['city']}，{a['state']}，{a['country']}；{postcode}。"
            )
        postcode = f"zip {a['zip']}" if "zip" in a else "zip is not provided; please ask me"
        return (
            f"Change order {oid} to {a['address1']}, address line 2 {line2 or 'empty'}, "
            f"{a['city']}, {a['state']}, {a['country']}; {postcode}."
        )
    items = ", ".join(item_list(a)) if "item_ids" in a else None
    method = a.get("payment_method_id")
    if language == "zh":
        chosen = f"只退商品{items}" if items else "具体退哪些商品尚未决定，请先询问，不能替我挑"
        refund = (
            f"退款方式选我的现有支付标识{method}"
            if method
            else "退款方式还未决定，请先询问，不要代选"
        )
        return f"订单{oid}要申请退货；{chosen}；{refund}。"
    chosen = (
        f"return only item IDs {items}" if items else "items are undecided; ask me before choosing"
    )
    refund = (
        f"use my existing refund method {method}"
        if method
        else "refund method is undecided; ask me"
    )
    return f"Request a return for order {oid}; {chosen}; {refund}."


def delta_text(before: Intent, after: Intent, language):
    if before.tool_name != after.tool_name:
        prefix = (
            "前一方案尚未确认，现在改为："
            if language == "zh"
            else ("The previous proposal is not approved. Instead: ")
        )
        return prefix + describe(after, language)
    labels = {
        "zh": {
            "reason": "取消原因",
            "item_ids": "退货商品清单",
            "payment_method_id": "退款方式",
            "address1": "街道",
            "address2": "地址第二行",
            "city": "城市",
            "state": "州",
            "country": "国家",
            "zip": "邮编",
        },
        "en": {
            "reason": "cancellation reason",
            "item_ids": "return item IDs",
            "payment_method_id": "refund method",
            "address1": "street",
            "address2": "address line 2",
            "city": "city",
            "state": "state",
            "country": "country",
            "zip": "zip",
        },
    }
    fields = []
    for key, value in after.arguments.items():
        if before.arguments.get(key) == value:
            continue
        if isinstance(value, list):
            value = ", ".join(str(item) for item in value)
        if value == "":
            value = "清空" if language == "zh" else "empty"
        fields.append(f"{labels[language][key]}={value}")
    if language == "zh":
        return "尚未确认，请更正：" + "；".join(fields) + "。未修改字段沿用上一方案。"
    return "Before approval, correct " + "; ".join(fields) + ". Keep all other fields unchanged."


class Author:
    """材料作者工具，生成完整阶段；运行时模拟器不使用此类。"""

    def __init__(self, *, split, index, user, initial):
        self.task_id = f"e3v2-{split}-{index:02}"
        self.split, self.index, self.user = split, index, user
        self.language = "en" if index in {9, 10, 12} else "zh"
        self.switch = index in {11, 12}
        self.initial_language = self.language
        self.initial, self.current = initial, initial
        intro = (
            f"你好，我的邮箱是{user.email}。"
            if self.language == "zh"
            else (f"Hello, my email is {user.email}. ")
        )
        self.message = intro + describe(initial, self.language)
        self.stages = []

    def lang(self):
        if self.switch:
            self.language = "en" if self.language == "zh" else "zh"
        return self.language

    def add(self, trigger, target, text, language, slot=None):
        update = FactUpdate(
            tool_name=target.tool_name if target.tool_name != self.current.tool_name else None,
            arguments=target.arguments
            if target.tool_name != self.current.tool_name
            else {
                k: v
                for k, v in target.arguments.items()
                if k not in self.current.arguments or self.current.arguments[k] != v
            },
        )
        self.stages.append(
            Stage.model_validate(
                {
                    "trigger": trigger,
                    "label": f"{self.split}-step-{len(self.stages) + 1}",
                    "expected": self.current.model_dump(mode="json")
                    if trigger == "proposal"
                    else None,
                    "slot": slot,
                    "reply": text,
                    "reply_language": language,
                    "update": update.model_dump(mode="json"),
                }
            )
        )
        self.current = target
        return self

    def change(self, target, *, full=False):
        language = self.lang()
        text = describe(target, language) if full else delta_text(self.current, target, language)
        return self.add("proposal", target, text, language)

    def ask(self, slot, target):
        language = self.lang()
        field = SLOT_FIELD[slot]
        value = target.arguments[field]
        if isinstance(value, list):
            value = ", ".join(str(item) for item in value)
        if language == "zh":
            text = f"补充确认：{slot}={value}，其余按刚才要求。"
        else:
            text = f"My choice is {slot}={value}; keep the other details as requested."
        return self.add("question", target, text, language, slot)

    def yes(self):
        language = self.lang()
        text = (
            "好的，就按刚才方案办。" if language == "zh" else "Okay, go ahead with that proposal."
        )
        return self.add("proposal", self.current, text, language)

    def refuse(self, target):
        language = self.lang()
        prefix = (
            "明白不能直接取消，那改为："
            if language == "zh"
            else ("Understood that cancellation is unavailable. Instead: ")
        )
        return self.add("refusal", target, prefix + describe(target, language), language)

    def finish(self):
        return UserScript.model_validate(
            {
                "task_id": self.task_id,
                "source_user_id": self.user.user_id,
                "source_task": "new-holdout-combination-20261027",
                "family": self.split + "-" + NAMES[self.split][self.index - 1],
                "title": NAMES[self.split][self.index - 1],
                "initial_language": self.initial_language,
                "known_identity": {"email": self.user.email},
                "initial_intent": self.initial.model_dump(mode="json"),
                "stages": [s.model_dump(mode="json") for s in self.stages],
                "final_intent": self.current.model_dump(mode="json"),
                "max_events": 12,
            }
        )


def plan_for(split, index, user, order):
    oid = order.order_id
    a = intent(
        ADDRESS,
        {
            "order_id": oid,
            "address1": f"{840 + index} Elm Street",
            "address2": "Suite 3",
            "city": "Seattle",
            "state": "WA",
            "country": "USA",
            "zip": "98101",
        },
    )
    c = intent(CANCEL, {"order_id": oid, "reason": "no longer needed"})
    r = intent(
        RETURN,
        {
            "order_id": oid,
            "item_ids": [order.items[0].item_id],
            "payment_method_id": order.payment_history[0].payment_method_id,
        },
    )
    gift = next(
        (
            key
            for key, method in user.payment_methods.items()
            if method.source == "gift_card" and key != r.arguments["payment_method_id"]
        ),
        None,
    )
    ids = [item.item_id for item in order.items[:3]]

    def changed(value, **updates):
        return intent(value.tool_name, value.arguments | updates)

    def missing(value, slot):
        args = dict(value.arguments)
        args.pop(SLOT_FIELD[slot])
        return intent(value.tool_name, args)

    def author(initial):
        return Author(split=split, index=index, user=user, initial=initial)

    if split == "validation":
        if index == 1:
            r2 = changed(r, item_ids=ids[:2])
            x = author(r2).change(c).refuse(r)
        elif index == 2:
            x = (
                author(a)
                .change(missing(c, "reason"))
                .ask("reason", c)
                .change(changed(c, reason="ordered by mistake"))
            )
        elif index == 3:
            x = author(c).change(missing(a, "zip")).ask("zip", a).change(changed(a, address2=""))
        elif index == 4:
            r2 = changed(r, item_ids=ids[:2])
            paid = changed(r2, payment_method_id=gift)
            x = (
                author(missing(r2, "items"))
                .ask("items", r2)
                .change(paid)
                .change(changed(paid, item_ids=[ids[1]]))
            )
        elif index == 5:
            x = author(a).yes().change(changed(a, city="Portland", state="OR", zip="97205"))
        elif index == 6:
            x = author(c).yes().change(changed(c, reason="ordered by mistake"))
        elif index == 7:
            x = (
                author(changed(r, item_ids=ids[:2]))
                .yes()
                .change(changed(r, item_ids=ids[:2], payment_method_id=gift))
            )
        elif index == 8:
            x = (
                author(missing(a, "zip"))
                .ask("zip", a)
                .change(changed(a, address1="910 Oak Road", address2=""))
                .yes()
            )
        elif index == 9:
            x = author(missing(c, "reason")).ask("reason", c).change(a)
        elif index == 10:
            x = (
                author(missing(r, "payment"))
                .ask("payment", changed(r, payment_method_id=gift))
                .change(changed(r, payment_method_id=gift, item_ids=[ids[1]]))
            )
        elif index == 11:
            x = author(a).change(c).change(changed(a, address1="950 Spruce Road", address2=""))
        else:
            x = (
                author(r)
                .change(changed(r, item_ids=ids[:2]))
                .yes()
                .change(changed(r, item_ids=ids[:2], payment_method_id=gift))
            )
    else:
        if index == 1:
            x = (
                author(missing(c, "reason"))
                .ask("reason", c)
                .change(missing(a, "zip"))
                .ask("zip", a)
                .change(changed(a, city="Tacoma", zip="98402", address2=""))
            )
        elif index == 2:
            x = author(c).refuse(missing(r, "items")).ask("items", r).yes()
        elif index == 3:
            paid = changed(r, payment_method_id=gift)
            x = (
                author(r)
                .change(paid)
                .change(changed(paid, item_ids=ids[:2]))
                .change(changed(paid, item_ids=[ids[2]]))
            )
        elif index == 4:
            x = (
                author(missing(a, "zip"))
                .ask("zip", a)
                .change(missing(c, "reason"))
                .ask("reason", c)
            )
        elif index == 5:
            x = author(c).change(a).change(changed(c, reason="ordered by mistake"))
        elif index == 6:
            r2 = changed(r, item_ids=ids[:2])
            x = (
                author(missing(r2, "payment"))
                .ask("payment", r2)
                .change(changed(r, item_ids=[ids[1]]))
                .change(changed(r, item_ids=[ids[1]], payment_method_id=gift))
            )
        elif index == 7:
            clear = changed(a, address2="")
            x = author(a).change(clear).change(changed(a, address2="Apt 12")).yes()
        elif index == 8:
            corrected = changed(c, reason="ordered by mistake")
            x = author(c).change(corrected).yes().change(a)
        elif index == 9:
            x = (
                author(missing(r, "items"))
                .ask("items", r)
                .change(changed(r, payment_method_id=gift))
                .yes()
            )
        elif index == 10:
            x = (
                author(missing(a, "zip"))
                .ask("zip", a)
                .change(
                    changed(
                        a,
                        address1="888 Ash Avenue",
                        address2="",
                        city="Boston",
                        state="MA",
                        zip="02108",
                    ),
                    full=True,
                )
                .yes()
            )
        elif index == 11:
            x = (
                author(missing(r, "items"))
                .ask("items", r)
                .change(changed(r, payment_method_id=gift))
                .change(changed(r, payment_method_id=gift, item_ids=[ids[1]]))
            )
        else:
            corrected = changed(c, reason="ordered by mistake")
            x = (
                author(missing(c, "reason"))
                .ask("reason", c)
                .change(corrected)
                .change(missing(a, "zip"))
                .ask("zip", changed(a, zip="98109"))
            )
    return x, x.finish()


def build(root: Path):
    dev, dev_manifest = load_dev(root)
    snapshot = load_snapshot(root)
    used = set()
    for rel in (
        "data/tasks/m0_dev_reviews.jsonl",
        "data/tasks/m3/user_scripts.jsonl",
        "data/tasks/nl_v1/user_scripts.jsonl",
        "data/tasks/private/e3_dev_v1/user_scripts.jsonl",
    ):
        for line in (root / rel).read_text("utf-8").splitlines():
            row = json.loads(line)
            used.add(
                row.get(
                    "user_id", row.get("source_user_id", row.get("plan", {}).get("source_user_id"))
                )
            )
    used |= {case["script"].source_user_id for case in dev}
    excluded = sorted(used)
    orders = sorted(
        snapshot.orders.values(), key=lambda order: fingerprint(["e3-v2-holdout", order.order_id])
    )
    signatures = {signature(case["script"]): case["public"].task_id for case in dev}
    files, groups, reviews = {}, {}, []
    for split in ("validation", "test"):
        inputs, scripts, criteria = [], [], []
        return_indexes = {1, 4, 7, 10, 12} if split == "validation" else {2, 3, 6, 9, 11}
        gift_indexes = {4, 7, 10, 12} if split == "validation" else {3, 6, 9, 11}
        for index in range(1, 13):

            def eligible(
                order, index=index, return_indexes=return_indexes, gift_indexes=gift_indexes
            ):
                if order.user_id in used or order.status != (
                    "delivered" if index in return_indexes else "pending"
                ):
                    return False
                if len({item.item_id for item in order.items[:3]}) < 3:
                    return False
                return index not in gift_indexes or any(
                    method.source == "gift_card"
                    and key != order.payment_history[0].payment_method_id
                    for key, method in snapshot.users[order.user_id].payment_methods.items()
                )

            order = next(order for order in orders if eligible(order))
            used.add(order.user_id)
            author, script = plan_for(split, index, snapshot.users[order.user_id], order)
            sig = signature(script)
            if sig in signatures:
                raise ValueError(
                    f"Combination signature collision: {script.task_id} vs {signatures[sig]}"
                )
            signatures[sig] = script.task_id
            for stage in script.stages:
                if stage.expected:
                    ARGUMENTS[stage.expected.tool_name].model_validate(stage.expected.arguments)
            ARGUMENTS[script.final_intent.tool_name].model_validate(script.final_intent.arguments)
            kind: Literal["cancel", "address", "return"] = (
                "cancel"
                if script.final_intent.tool_name == CANCEL
                else "address"
                if script.final_intent.tool_name == ADDRESS
                else "return"
            )
            goal = CaseCriteria(
                task_id=script.task_id,
                kind=kind,
                user_id=order.user_id,
                order_id=order.order_id,
                arguments=script.final_intent.arguments,
            )
            inputs.append({"task_id": script.task_id, "first_user_message": author.message})
            scripts.append(script.model_dump(mode="json"))
            criteria.append(goal.model_dump(mode="json"))
            reviews.append(
                {
                    "task_id": script.task_id,
                    "split": split,
                    "signature": sig,
                    "user_id": order.user_id,
                    "family": script.family,
                    "initial_known": script.initial_intent.model_dump(mode="json"),
                    "stage_updates": [s.update.model_dump(mode="json") for s in script.stages],
                    "reviewer": "assistant author + structural audit; not independent annotation",
                }
            )
        groups[split] = {
            "tasks": 12,
            "language_groups": {"zh_only": 8, "en_only": 2, "language_switch": 2},
            "families": [script["family"] for script in scripts],
        }
        for name, rows in (("inputs", inputs), ("user_scripts", scripts), ("criteria", criteria)):
            rel = (
                f"data/tasks/e3_v2_{split}/inputs.jsonl"
                if name == "inputs"
                else f"data/tasks/private/e3_v2_{split}/{name}.jsonl"
            )
            files[rel] = "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
            ).encode()
    files["data/tasks/private/e3_v2_holdout/author_reviews.jsonl"] = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in reviews
    ).encode()
    manifest = {
        "suite_id": "e3-v2-holdout-v1",
        "created_on": "2026-10-27",
        "protocol_sha256": sha(root / "configs/evaluation/e3_protocol_v2.json"),
        "dev_manifest_sha256": sha(root / "data/manifests/e3_v2_dev_suite.json"),
        "dev_files": dev_manifest["files"],
        "groups": groups,
        "excluded_users": excluded,
        "files": {rel: hashlib.sha256(body).hexdigest() for rel, body in files.items()},
        "signature_rule": (
            "tools, triggers/slots, changed/cleared fields and item-set operations; "
            "ignore actual IDs, languages and wording"
        ),
        "current_run_policy": (
            "sealed; no live validation/test before candidate selection and freeze"
        ),
        "limitations": [
            "Assistant-authored and reviewed; shared retail domain and workflow primitives",
            "Signature separation is a structural audit, not proof of distribution independence",
            "Offline replay validates the driver, not held-out real-model performance",
        ],
    }
    files["data/manifests/e3_v2_holdout_v1.json"] = (
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    ).encode()
    for rel, body in files.items():
        path = root / rel
        if path.exists() and path.read_bytes() != body:
            raise ValueError("Holdout materials differ; create a new suite version")
    for rel, body in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(body)
    return manifest


def load(root: Path, split: str):
    if split not in {"validation", "test"}:
        raise ValueError("Use the original loader for development")
    manifest = json.loads((root / "data/manifests/e3_v2_holdout_v1.json").read_text("utf-8"))
    if sha(root / "configs/evaluation/e3_protocol_v2.json") != manifest["protocol_sha256"]:
        raise ValueError("Holdout protocol mismatch")
    if sha(root / "data/manifests/e3_v2_dev_suite.json") != manifest["dev_manifest_sha256"]:
        raise ValueError("Development manifest changed")
    tables = {}
    for rel, digest in manifest["files"].items():
        path = (root / rel).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path) != digest:
            raise ValueError("Holdout file hash mismatch")
        if f"e3_v2_{split}/" in rel:
            rows = [json.loads(s) for s in path.read_text("utf-8").splitlines()]
            tables[path.name] = {row["task_id"]: row for row in rows}
            if len(tables[path.name]) != len(rows):
                raise ValueError("Duplicate holdout IDs")
    public, plans, goals = [
        tables[name] for name in ("inputs.jsonl", "user_scripts.jsonl", "criteria.jsonl")
    ]
    if not public.keys() == plans.keys() == goals.keys() or len(public) != 12:
        raise ValueError("Holdout task mismatch")
    return [
        {
            "public": VisibleTaskInput.model_validate(public[task]),
            "script": UserScript.model_validate(plans[task]),
            "criteria": CaseCriteria.model_validate(goals[task]),
        }
        for task in public
    ], manifest
