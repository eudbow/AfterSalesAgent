"""Materialize ten reviewed train-only samples; never turn hidden answers into Agent input."""

from __future__ import annotations

import json
from pathlib import Path

from aftersalesagent.business.source_audit import verify_source

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/upstream/tau2-bench/data/tau2/domains/retail"
COMMIT = "fc0055dc4e0a316c3f83133267fbd6faaa770992"
# 每项是逐条阅读需求、政策、全部用户订单后的审阅决定；不是按参考动作盲筛。
REVIEWS = [
    (
        "10",
        "mia_garcia_4516",
        ["return", "handoff"],
        "mia-return",
        "I want to return both of my orders, refunding each to the other order's payment method.",
        [
            "两单均已送达，原付款方式分别为信用卡与 PayPal，交叉退款不符合原方式/已有礼品卡条件。",
            "解释限制后按用户要求转人工；不得修改订单。",
        ],
    ),
    (
        "11",
        "mia_garcia_4516",
        ["return", "clarification"],
        "mia-return",
        "I want to return both of my orders, refunding each to the other order's payment method.",
        [
            "与任务10同实体/需求家族，只用于开发，不当作独立泛化证据。",
            "用户接受原付款方式后，对两单分别确认并提交退货请求。",
        ],
    ),
    (
        "16",
        "fatima_johnson_7581",
        ["cancel", "return", "query"],
        "fatima-mixed",
        (
            "Please cancel all my pending orders and return only the watch I "
            "received. How much can I get back?"
        ),
        [
            "两单 pending 可取消；已送达订单仅退指定手表。",
            "金额依据成交价：3131.10 + 4777.75 + 367.38 = 8276.23；只模拟退款记录。",
        ],
    ),
    (
        "24",
        "sofia_hernandez_5364",
        ["cancel", "withdraw_confirmation", "query"],
        "sofia-withdraw",
        "I would like to cancel my grill order.",
        [
            "必须按姓名加邮编98193定位，不能仅凭同名用户匹配。",
            "用户在确认时撤销取消；数据库应保持不变。",
            "之后查询另一单两件T恤的材质：棉与聚酯。完整后续意图留在用户模拟器材料。",
        ],
    ),
    (
        "48",
        "daiki_johnson_9523",
        ["return", "query"],
        "daiki-return",
        (
            "Please return the air purifier I received to my original payment "
            "method. Also check whether I can return the vacuum cleaner, but do "
            "not process it."
        ),
        [
            "原单已送达，只提交空气净化器9534205511的退货申请。",
            "吸尘器仅查询资格；不能将询问理解成确认执行。",
        ],
    ),
    (
        "59",
        "yusuf_taylor_7149",
        ["cancel", "address", "query"],
        "yusuf-address",
        (
            "Please check orders #W2702727 and #W8268610. I want to cancel the "
            "older one if you cannot guarantee processing within five days, and "
            "change #W2702727 to 1234 Elm St, Springfield, IL 62701."
        ),
        [
            "两单均 pending；取消#W8268610并修改#W2702727地址，分别确认。",
            "数据没有真实下单日期与交付保证字段，不编造日期或保证；用户提出的要求属于用户陈述。",
            "订单金额625.60，取消退款164.28；地址修改不得改变商品与金额。",
        ],
    ),
    (
        "57",
        "ivan_hernandez_6923",
        ["query", "cancel", "policy_denial"],
        "ivan-refund-condition",
        (
            "When will order W4284542 arrive? If it has not shipped, cancel the "
            "air purifier; if you must cancel the whole order, only do so if the "
            "refund can go to my gift card."
        ),
        [
            "原单pending，付款方式是信用卡；取消必须退回原方式，不能改为礼品卡，因此不执行取消。",
            "没有承诺到货时间的数据；仅报告可核验状态，不编造日期。",
            "参考动作列表为空只表示不修改数据库，应用仍需身份定位、查询与解释。",
        ],
    ),
    (
        "73",
        "fatima_wilson_7472",
        ["return"],
        "fatima-except-coffee",
        "I want to return everything I just bought except the coffee machine.",
        [
            "已送达订单有五件商品，排除咖啡机7441167885后退其余四件。",
            "参考轨迹没有身份查询，但应用仍必须独立满足政策的身份与确认要求。",
        ],
    ),
    (
        "84",
        "chen_silva_7485",
        ["return", "change_confirmation"],
        "chen-tablets",
        "I received two tablets and want to return the less expensive one to my credit card.",
        [
            "两个平板在不同订单中，价格938.92与989.70；需要查看用户全部订单，不能只看目标单。",
            "用户在确认时改为较贵平板6065192424并退款至已有礼品卡；旧确认失效。",
            "参考目标只修改#W9571698，其他商品与另一平板保留。",
        ],
    ),
    (
        "113",
        "yara_muller_8652",
        ["cancel", "clarification"],
        "yara-cancel",
        "Please cancel all my pending orders.",
        [
            "两单 pending 可取消；必须询问取消原因，用户回答ordered by mistake后才确认执行。",
            "原动作ID前缀为114而任务ID为113；保留原资料，使用task.id关联，不按前缀猜任务。",
        ],
    ),
]


def main() -> None:
    errors = verify_source(ROOT)
    if errors:
        raise SystemExit("\n".join(errors))
    tasks = {t["id"]: t for t in json.loads((SOURCE / "tasks.json").read_text("utf-8"))}
    splits = json.loads((SOURCE / "split_tasks.json").read_text("utf-8"))
    db = json.loads((SOURCE / "db.json").read_text("utf-8"))
    public, reviews = [], []
    for tid, uid, tags, family, first_message, notes in REVIEWS:
        if tid not in splits["train"]:
            raise ValueError(f"Not a train task: {tid}")
        task = tasks[tid]
        private_rel = f"data/tasks/private/m0/task_{tid}.json"
        private = {
            "source_task_id": tid,
            "source_commit": COMMIT,
            "initial_fixture": "data/upstream/tau2-bench/data/tau2/domains/retail/db.json",
            "initial_state": task["initial_state"],
            "user_simulator": task["user_scenario"],
            "evaluation_criteria": task["evaluation_criteria"],
            "access_boundary": (
                "user_simulator and evaluation materials; never pass this file to Agent"
            ),
        }
        path = ROOT / private_rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(private, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        reviews.append(
            {
                "sample_id": f"m0-retail-{tid}",
                "source_task_id": tid,
                "source_split": "train",
                "development_only": True,
                "business_tags": tags,
                "family_id": family,
                "user_id": uid,
                "relevant_order_ids": db["users"][uid]["orders"],
                "review_status": "reviewed",
                "review_notes": notes,
                "private_fixture": private_rel,
                "policy_rule_ids": ["R01", "R02", "R03"]
                + (["R04"] if "cancel" in tags else [])
                + (["R05"] if "address" in tags else [])
                + (["R06"] if "return" in tags else []),
                "source_commit": COMMIT,
            }
        )
        public.append({"task_id": f"m0-retail-{tid}", "first_user_message": first_message})
    for name, rows in (("m0_dev_inputs.jsonl", public), ("m0_dev_reviews.jsonl", reviews)):
        (ROOT / "data/tasks" / name).write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
        )
    selection = {
        "sample_count": 10,
        "source_commit": COMMIT,
        "source_task_ids": [r[0] for r in REVIEWS],
        "upstream_split": "train",
        "purpose": "reviewed development examples, not benchmark results",
        "inputs_adapted": True,
        "evaluation_criteria_preserved": True,
        "future_split_rule": (
            "Group shared entity and intent families; never split a family by task ID alone."
        ),
        "upstream_test_ids_used": [],
    }
    (ROOT / "data/manifests/sample_selection.json").write_text(
        json.dumps(selection, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("Prepared 10 reviewed train-only samples; Agent inputs exclude private fields.")


if __name__ == "__main__":
    main()
