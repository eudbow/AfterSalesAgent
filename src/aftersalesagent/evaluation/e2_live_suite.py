"""E2八场景集成验收；公开消息、用户后续脚本与私有判分分开冻结。"""

import hashlib
import json
from pathlib import Path

from aftersalesagent.evaluation.e1_suite import load as load_knowledge
from aftersalesagent.evaluation.nl_suite import load as load_nl


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode()


def build(root: Path):
    cases, _ = load_nl(root, "dev")
    by_id = {c.public.task_id: c for c in cases}
    public, plans, criteria = [], [], []
    definitions = [
        ("order_product", "nl1-dev-01"),
        ("cancel", "nl1-dev-04"),
        ("address_correction", "nl1-dev-08"),
        ("return_missing_item", "nl1-dev-02"),
        ("plain_yes", "nl1-dev-10"),
        ("invalid_state", "nl1-dev-11"),
        ("policy", None),
        ("proposal_response_loss", "nl1-dev-04"),
    ]
    for i, (scenario, source_id) in enumerate(definitions, 1):
        task_id = f"e2live-{i:02}"
        if source_id is None:
            source = load_knowledge(root, "dev")[0][2]
            public.append(
                {
                    "task_id": task_id,
                    "scenario": scenario,
                    "kind": "knowledge",
                    "question": source["public"]["question"],
                    "filters": source["public"]["filters"],
                }
            )
            plans.append({"task_id": task_id, "plan": None, "extra": {}})
            criteria.append(
                {
                    "task_id": task_id,
                    "criteria": source["criteria"],
                    "extra": {},
                    "source_task": "e1-dev-03",
                }
            )
            continue
        source = by_id[source_id]
        message = source.public.first_user_message
        plan = source.plan.model_dump(mode="json")
        goal = source.criteria.model_dump(mode="json")
        plan["task_id"] = goal["task_id"] = task_id
        extra = {}
        if scenario == "order_product":
            # 选定源订单第一件商品，公开给客户的问题，不从隐藏答案动态补给模型。
            from aftersalesagent.business.source_audit import load_snapshot

            snapshot = load_snapshot(root)
            product = snapshot.orders[source.criteria.order_id].items[0].product_id
            message += f"另外请查询其中商品{product}的可选规格，回复保留订单号和商品编号。"
            extra["product_id"] = product
        if scenario == "return_missing_item":
            from aftersalesagent.business.source_audit import load_snapshot

            snapshot = load_snapshot(root)
            customer = snapshot.users[source.plan.source_user_id]
            assert len(snapshot.orders[source.plan.order_id].items) > 1
            message = (
                f"你好，我的邮箱是{customer.email}。我想退订单{source.plan.order_id}里的部分商品，"
                "具体退哪一件我还没告诉你。"
                f"退款请用原支付方式{source.plan.final_arguments['payment_method_id']}。"
            )
            item = source.plan.final_arguments["item_ids"][0]
            extra["item_answer"] = f"我只退商品编号{item}，订单里的其他商品都保留。"
        public.append(
            {
                "task_id": task_id,
                "scenario": scenario,
                "kind": "business",
                "first_user_message": message,
            }
        )
        plans.append({"task_id": task_id, "plan": plan, "extra": extra})
        criteria.append(
            {"task_id": task_id, "criteria": goal, "extra": extra, "source_task": source_id}
        )
    contents = {
        "data/tasks/e2_live_v1/inputs.jsonl": b"".join(encoded(v) for v in public),
        "data/tasks/private/e2_live_v1/user_scripts.jsonl": b"".join(encoded(v) for v in plans),
        "data/tasks/private/e2_live_v1/criteria.jsonl": b"".join(encoded(v) for v in criteria),
    }
    for rel, content in contents.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_bytes() != content:
            raise ValueError("E2 live frozen data differ; create a new version")
        path.write_bytes(content)
    manifest = {
        "suite_id": "e2-live-integration-v1",
        "frozen_on": "2026-10-25",
        "scenarios": 8,
        "episodes": 16,
        "smoke_task": "e2live-01",
        "paired_order": "alternate native/mcp first by task index",
        "files": {p: hashlib.sha256(c).hexdigest() for p, c in contents.items()},
        "limitations": [
            (
                "Known development scenarios reused/adapted; integration check, "
                "not fresh held-out accuracy"
            ),
            "Seven business scenarios and one separate read-only policy consultation",
            "No automatic task reruns; all outcomes retained; no Skill loaded",
        ],
    }
    path = root / "data/manifests/e2_live_v1_suite.json"
    content = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    if path.exists() and path.read_bytes() != content:
        raise ValueError("E2 live manifest already frozen")
    path.write_bytes(content)
    return manifest


def load(root: Path):
    manifest = json.loads((root / "data/manifests/e2_live_v1_suite.json").read_text("utf-8"))
    for rel, sha in manifest["files"].items():
        if hashlib.sha256((root / rel).read_bytes()).hexdigest() != sha:
            raise ValueError("E2 live task hash mismatch")

    def rows(path):
        return [json.loads(line) for line in (root / path).read_text("utf-8").splitlines()]

    public = rows("data/tasks/e2_live_v1/inputs.jsonl")
    plans = {v["task_id"]: v for v in rows("data/tasks/private/e2_live_v1/user_scripts.jsonl")}
    criteria = {v["task_id"]: v for v in rows("data/tasks/private/e2_live_v1/criteria.jsonl")}
    cases = [
        {"public": p, "user_script": plans[p["task_id"]], "private": criteria[p["task_id"]]}
        for p in public
    ]
    assert len(cases) == 8
    return cases, manifest
