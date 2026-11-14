"""六个已见事实上的开发诊断，非新留出集；S1固定Skill，两次预定重复。"""

from __future__ import annotations

import json
from pathlib import Path

from aftersalesagent.evaluation.e3_v2.contracts import UserScript
from aftersalesagent.evaluation.e3_v2.suite import load
from aftersalesagent.schemas.contracts import VisibleTaskInput


def cases(root: Path):
    old, _ = load(root)
    result = []
    for index, (initial_language, later_language) in enumerate(
        (("zh", "zh"), ("en", "en"), ("en", "zh"), ("zh", "en")), 1
    ):
        source = old[11]
        raw = source["script"].model_dump(mode="json")
        raw["task_id"] = task_id = f"e3diag-language-{index:02}"
        raw["initial_language"] = initial_language
        raw["family"] = "development_language_observation_" + str(index)
        raw["title"] = f"同一业务下{initial_language}到{later_language}缺邮编追问"
        raw["stages"][0]["reply_language"] = later_language
        raw["stages"][1]["reply_language"] = later_language
        a = raw["stages"][0]["update"]["arguments"]
        raw["stages"][0]["reply"] = (
            f"先别取消。改为修改地址：{a['address1']}，第二行{a['address2']}，"
            f"{a['city']}，{a['state']}，{a['country']}。新邮编尚未提供，请先问我。"
            if later_language == "zh"
            else f"Do not cancel. Change the address to {a['address1']}, line 2 {a['address2']}, "
            f"{a['city']}, {a['state']}, {a['country']}. I have not supplied the new zip. Ask me."
        )
        zip_code = raw["stages"][1]["update"]["arguments"]["zip"]
        raw["stages"][1]["reply"] = (
            f"新邮编是{zip_code}，其他地址字段不变。"
            if later_language == "zh"
            else f"The new zip is {zip_code}; keep the other address fields."
        )
        email, oid = raw["known_identity"]["email"], raw["initial_intent"]["arguments"]["order_id"]
        message = (
            f"你好，我的邮箱是{email}。请取消订单{oid}，原因是不再需要（no longer needed）。"
            if initial_language == "zh"
            else f"Hello, my email is {email}. Cancel order {oid}; the reason is no longer needed."
        )
        result.append(
            {
                "public": VisibleTaskInput(task_id=task_id, first_user_message=message),
                "script": UserScript.model_validate(raw),
                "criteria": source["criteria"].model_copy(update={"task_id": task_id}),
            }
        )
    for index, language in enumerate(("zh", "en"), 1):
        source = old[3]
        raw = source["script"].model_dump(mode="json")
        task_id = f"e3diag-preparation-{index:02}"
        raw["task_id"], raw["initial_language"] = task_id, language
        raw["family"] = "development_preparation_permission_" + language
        raw["title"] = "部分退货准备卡片流程诊断_" + language
        for stage in raw["stages"]:
            stage["reply_language"] = language
        initial = raw["initial_intent"]["arguments"]
        ids = raw["stages"][0]["update"]["arguments"]["item_ids"]
        if language == "en":
            raw["stages"][0]["reply"] = f"Return only item IDs {ids[0]} and {ids[1]}."
            raw["stages"][1]["reply"] = f"Do not confirm yet. Keep {ids[0]}; return only {ids[1]}."
            message = (
                f"My email is {raw['known_identity']['email']}. I want a partial return "
                f"of order {initial['order_id']}. Ask me which items; do not select for me. "
                f"Refund to the original method {initial['payment_method_id']}."
            )
        else:
            message = source["public"].first_user_message
        result.append(
            {
                "public": VisibleTaskInput(task_id=task_id, first_user_message=message),
                "script": UserScript.model_validate(raw),
                "criteria": source["criteria"].model_copy(update={"task_id": task_id}),
            }
        )
    return result


def describe(root: Path):
    values = cases(root)
    return {
        "tasks": 6,
        "repetitions": 2,
        "episodes": 12,
        "group": "S1",
        "maximum_model_requests": 144,
        "smoke_episodes_included": 1,
        "meaning": "Known development facts; not heldout evidence",
        "tasks_snapshot": [
            {
                "public": case["public"].model_dump(mode="json"),
                "script": case["script"].model_dump(mode="json"),
                "criteria": case["criteria"].model_dump(mode="json"),
            }
            for case in values
        ],
        "holdout_real_episodes": 0,
        "analysis_plan": (
            "Report all outcomes; preparation-question counts, permission "
            "branches, missing-card signals; "
            "language by current actual user and after-switch opportunities, "
            "blind explicit review. "
            "No S1 gain claim and no selective reruns."
        ),
    }


def source_audit(root: Path) -> dict:
    baseline = json.loads((root / "configs/baselines/e3_v2_s0_s1_dev_v1.json").read_text("utf-8"))
    return {
        "core_policy_has_explicit_english_only": False,
        "audit_method": "Inspected actual system messages, application policy and fixed Skill",
        "fixed_skill_has_current_user_language_rule": True,
        "core_policy_language": "Mostly English; influence is a hypothesis, not causal proof",
        "historical_protocol": baseline["frozen_config"]["protocol"]["version"],
        "historical_scores_preserved": True,
    }
