"""E3 新开发任务：组合情形，独立用户脚本和评分；不构建或查看新留出集。"""

import hashlib
import json
from pathlib import Path

from aftersalesagent.business.source_audit import load_snapshot, verify_source
from aftersalesagent.evaluation.suite import CaseCriteria
from aftersalesagent.schemas.contracts import VisibleTaskInput

SUITE = "e3-workflow-dev-v1"
TITLES = [
    "街道与第二行同时改正",
    "补邮编后跨城市修改",
    "两次改口且清空第二行",
    "先明确部分退货再缩减",
    "退货集合替换一件",
    "退款方式改为已有礼品卡",
    "补原因后再改原因",
    "取消改为修改地址",
    "改址改为取消",
    "已送达取消被拒后改退货",
    "补退款方式后普通文字同意",
    "切换意图后再补缺失邮编",
]
FAMILIES = [
    "address_joint_override",
    "address_clarify_then_override",
    "address_two_corrections_clear",
    "return_clarify_then_subset",
    "return_set_replacement",
    "return_payment_switch",
    "cancel_reason_reconsideration",
    "intent_cancel_to_address",
    "intent_address_to_cancel",
    "refusal_then_supported_return",
    "refund_choice_then_text_approval",
    "intent_switch_then_clarification",
]


def encode(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def build(root: Path):
    if problems := verify_source(root):
        raise ValueError("; ".join(problems))
    snapshot = load_snapshot(root)
    used = set()
    for rel in (
        "data/tasks/m0_dev_reviews.jsonl",
        "data/tasks/m3/user_scripts.jsonl",
        "data/tasks/nl_v1/user_scripts.jsonl",
    ):
        for line in (root / rel).read_text("utf-8").splitlines():
            row = json.loads(line)
            used.add(row.get("user_id", row.get("source_user_id")))
    excluded = sorted(used)
    orders = sorted(
        snapshot.orders.values(),
        key=lambda o: hashlib.sha256(("e3-dev-20261027:" + o.order_id).encode()).hexdigest(),
    )
    public, scripts, criteria = [], [], []
    for index, (title, family) in enumerate(zip(TITLES, FAMILIES, strict=True), 1):
        is_return = index in {4, 5, 6, 10, 11}

        def eligible(order, index=index, is_return=is_return):
            if order.user_id in used or order.status != ("delivered" if is_return else "pending"):
                return False
            if index in {4, 5} and len(order.items) < 3:
                return False
            return index != 6 or any(
                m.source == "gift_card" and key != order.payment_history[0].payment_method_id
                for key, m in snapshot.users[order.user_id].payment_methods.items()
            )

        order = next(o for o in orders if eligible(o))
        used.add(order.user_id)
        user = snapshot.users[order.user_id]
        oid = order.order_id
        task_id = f"e3dev-{index:02}"
        address = {
            "order_id": oid,
            "address1": f"{720 + index} Maple Street",
            "address2": "Unit 8",
            "city": "Seattle",
            "state": "WA",
            "country": "USA",
            "zip": "98101",
        }
        cancel = {"order_id": oid, "reason": "no longer needed"}
        ret = {
            "order_id": oid,
            "item_ids": [order.items[0].item_id],
            "payment_method_id": order.payment_history[0].payment_method_id,
        }
        steps = []

        def proposal(tool, args, reply, label, steps=steps):
            steps.append(
                {
                    "trigger": "proposal",
                    "expected_tool": tool,
                    "expected_arguments": dict(args),
                    "reply": reply,
                    "label": label,
                }
            )

        def question(slot, reply, label, steps=steps):
            steps.append({"trigger": "question", "slot": slot, "reply": reply, "label": label})

        def describe_address(args):
            return (
                f"{args['address1']}，第二行{args['address2'] or '留空'}，{args['city']}，"
                f"{args['state']}，{args['country']}，邮编{args['zip']}"
            )

        address_tool, cancel_tool, return_tool = (
            "modify_pending_order_address",
            "cancel_pending_order",
            "return_delivered_order_items",
        )
        final_kind, final_tool, final_args = "address", address_tool, dict(address)
        if index == 1:
            request = f"请把订单{oid}地址改为{describe_address(address)}。"
            proposal(
                address_tool,
                address,
                "街道更正为850 Birch Road，第二行请清空，其余保持不变。",
                "joint_correction",
            )
            final_args.update(address1="850 Birch Road", address2="")
        elif index == 2:
            request = (
                f"请将订单{oid}地址改成{address['address1']}，第二行Unit 8，Seattle，WA，USA。"
                "邮编尚未告诉你，请先询问，不要从旧订单推断。"
            )
            question("zip", "这次新地址邮编是98101。", "zip_answer")
            proposal(
                address_tool,
                address,
                "我改主意了：城市改Portland，州OR，邮编97205，其他不变。",
                "cross_state",
            )
            final_args.update(city="Portland", state="OR", zip="97205")
        elif index == 3:
            request = f"请将订单{oid}地址改为{describe_address(address)}。"
            proposal(
                address_tool,
                address,
                "先更正街道为900 Pine Avenue，其余字段不变。",
                "street_revision",
            )
            changed = dict(address, address1="900 Pine Avenue")
            proposal(
                address_tool,
                changed,
                "再更正：第二行清空，城市Tacoma，邮编98402；街道保留刚改的，其余不变。",
                "second_revision",
            )
            final_args = dict(changed, address2="", city="Tacoma", zip="98402")
        elif index in {4, 5}:
            a, b, c = [item.item_id for item in order.items[:3]]
            ret["item_ids"] = [a, b]
            if index == 4:
                request = (
                    f"想退订单{oid}的部分商品，具体哪些我还没说，请问我，不要替我挑。"
                    f"退款选原支付方式{ret['payment_method_id']}。"
                )
                question("items", f"我要退商品编号{a}和{b}，其他保留。", "item_choice")
                proposal(
                    return_tool,
                    ret,
                    f"先别确认：商品{a}不退了，只退{b}，退款方式不变。",
                    "shrink_set",
                )
                final_args = dict(ret, item_ids=[b])
            else:
                request = f"订单{oid}要退商品{a}和{b}，退款用原方式{ret['payment_method_id']}。"
                proposal(
                    return_tool,
                    ret,
                    (f"改一下退货清单：保留{a}不退，改为退{b}和{c}，其他商品不退，退款方式不变。"),
                    "replace_member",
                )
                final_args = dict(ret, item_ids=[b, c])
            final_kind, final_tool = "return", return_tool
        elif index == 6:
            gift = next(
                key
                for key, method in user.payment_methods.items()
                if method.source == "gift_card" and key != ret["payment_method_id"]
            )
            request = (
                f"订单{oid}只退商品{ret['item_ids'][0]}，"
                f"退款先选原支付方式{ret['payment_method_id']}。"
            )
            proposal(
                return_tool,
                ret,
                f"退款方式改为我已有的礼品卡{gift}，退货商品和订单不变。",
                "payment_revision",
            )
            final_kind, final_tool, final_args = (
                "return",
                return_tool,
                dict(ret, payment_method_id=gift),
            )
        elif index == 7:
            request = f"我想取消订单{oid}，取消原因还没决定，请先问我。"
            question("reason", "原因是我现在不需要了。", "reason_answer")
            proposal(
                cancel_tool,
                cancel,
                "更正一下取消原因：是我下错单了，不是不需要了。",
                "reason_revision",
            )
            final_kind, final_tool, final_args = (
                "cancel",
                cancel_tool,
                dict(cancel, reason="ordered by mistake"),
            )
        elif index == 8:
            request = f"我不需要订单{oid}了，请发起取消。"
            proposal(
                cancel_tool,
                cancel,
                f"先别取消，我还是要这个订单。请只改地址为{describe_address(address)}。",
                "cancel_to_address",
            )
        elif index == 9:
            request = f"请把订单{oid}地址改为{describe_address(address)}。"
            proposal(
                address_tool,
                address,
                "刚才改地址的要求作废，我已经不需要这个订单了，请改为取消。",
                "address_to_cancel",
            )
            final_kind, final_tool, final_args = "cancel", cancel_tool, cancel
        elif index == 10:
            request = f"订单{oid}我不想要了。即使已经送达，也请直接取消；若不符合规则请明确说明。"
            steps.append(
                {
                    "trigger": "refusal",
                    "label": "refusal_then_return",
                    "reply": (
                        f"明白，那改为退货：只退商品{ret['item_ids'][0]}，"
                        f"退款选原支付方式{ret['payment_method_id']}。"
                    ),
                }
            )
            final_kind, final_tool, final_args = "return", return_tool, ret
        elif index == 11:
            request = (
                f"我要退订单{oid}的商品{ret['item_ids'][0]}。"
                "退款方式尚未决定，请先询问，别替我选择原支付或礼品卡。"
            )
            question(
                "payment", f"退款方式选原支付方式{ret['payment_method_id']}。", "payment_answer"
            )
            proposal(return_tool, ret, "好的，就按刚才的退货方案办。", "plain_yes")
            final_kind, final_tool, final_args = "return", return_tool, ret
        else:
            request = f"不需要订单{oid}了，请帮我发起取消。"
            proposal(
                cancel_tool,
                cancel,
                (
                    f"先别取消，改成修改地址：{address['address1']}，"
                    "第二行Unit 8，Seattle，WA，USA。"
                    "新邮编我还没给你，请先问我。"
                ),
                "intent_switch",
            )
            question("zip", "新地址邮编98109，其他字段按上一条修改地址的要求。", "zip_after_switch")
            final_args.update(zip="98109")
        public.append(
            {
                "task_id": task_id,
                "first_user_message": f"你好，我的邮箱是{user.email}。{request}",
            }
        )
        scripts.append(
            {
                "task_id": task_id,
                "split": "dev",
                "family": family,
                "title": title,
                "source_user_id": user.user_id,
                "order_id": oid,
                "steps": steps,
                "final": {"tool_name": final_tool, "arguments": final_args},
                "max_events": 8,
            }
        )
        criteria.append(
            CaseCriteria(
                task_id=task_id,
                kind=final_kind,
                user_id=user.user_id,
                order_id=oid,
                arguments=final_args,
            ).model_dump(mode="json")
        )
    contents = {
        "data/tasks/e3_dev_v1/inputs.jsonl": b"".join(encode(x) for x in public),
        "data/tasks/private/e3_dev_v1/user_scripts.jsonl": b"".join(encode(x) for x in scripts),
        "data/tasks/private/e3_dev_v1/criteria.jsonl": b"".join(encode(x) for x in criteria),
    }
    manifest = {
        "suite_id": SUITE,
        "frozen_on": "2026-10-27",
        "split": "development_only",
        "tasks": 12,
        "repetitions": 2,
        "groups": ["S0", "S1"],
        "episodes": 48,
        "smoke_pair": {"task_id": "e3dev-01", "repetition": 1},
        "excluded_prior_users": excluded,
        "files": {p: hashlib.sha256(c).hexdigest() for p, c in contents.items()},
        "families": dict(zip(FAMILIES, TITLES, strict=True)),
        "primary": "business verifier + all scripted stages + no unconfirmed changes",
        "secondary": "argument errors, repeated invalid calls, language flags, usage and wall time",
        "order": "tasks then repetitions; alternate S0/S1 first by task index plus repetition",
        "limits_file": "configs/m3_live_limits.json",
        "limitations": [
            (
                "Assistant-authored development scenarios, "
                "not new validation/test or official benchmark"
            ),
            (
                "New users plus compositional scripts; "
                "same retail domain and related workflow semantics"
            ),
            "Two repetitions are descriptive, not sufficient statistical power",
            "No artificial error injection or production/Skill/budget edits during comparison",
            "Bilingual script triggers and coarse language flags require trace review",
        ],
    }
    contents["data/manifests/e3_dev_v1_suite.json"] = (
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    ).encode()
    for rel, body in contents.items():
        path = root / rel
        if path.exists() and path.read_bytes() != body:
            raise ValueError("E3 dev assets differ; use a new suite version")
    for rel, body in contents.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(body)
    return manifest


def load(root: Path):
    manifest = json.loads((root / "data/manifests/e3_dev_v1_suite.json").read_text("utf-8"))
    tables = {}
    for rel, digest in manifest["files"].items():
        path = (root / rel).resolve()
        if (
            not path.is_relative_to(root.resolve())
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest
        ):
            raise ValueError("E3 task hash mismatch")
        rows = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
        tables[path.name] = {row["task_id"]: row for row in rows}
    public, plans, goals = (
        tables[k] for k in ("inputs.jsonl", "user_scripts.jsonl", "criteria.jsonl")
    )
    if not (public.keys() == plans.keys() == goals.keys()) or len(public) != 12:
        raise ValueError("E3 task counts/IDs differ")
    users = []
    cases = []
    for task_id in public:
        VisibleTaskInput.model_validate(public[task_id])
        goal = CaseCriteria.model_validate(goals[task_id])
        plan = plans[task_id]
        if goal.user_id != plan["source_user_id"] or goal.arguments != plan["final"]["arguments"]:
            raise ValueError("User intent and acceptance target disagree")
        if plan["split"] != "dev" or goal.user_id in manifest["excluded_prior_users"]:
            raise ValueError("E3 user/split leakage")
        users.append(goal.user_id)
        cases.append({"public": public[task_id], "plan": plan, "criteria": goals[task_id]})
    if len(set(users)) != 12:
        raise ValueError("E3 development users must be distinct")
    return cases, manifest
