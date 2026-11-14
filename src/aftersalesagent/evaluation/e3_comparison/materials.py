"""三个集合事前绑定；留出加工只改变公开时序，最终业务目标保持原样。"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from aftersalesagent.evaluation.e3_candidates.diagnostics import cases as diagnostic_cases
from aftersalesagent.evaluation.e3_v2.contracts import (
    FactUpdate,
    UserScript,
    apply_update,
)
from aftersalesagent.evaluation.e3_v2.holdout import load as load_holdout
from aftersalesagent.evaluation.e3_v2.holdout import signature
from aftersalesagent.evaluation.e3_v2.suite import load as load_dev
from aftersalesagent.evaluation.e3_v2.suite import sha
from aftersalesagent.evaluation.suite import CaseCriteria
from aftersalesagent.schemas.contracts import VisibleTaskInput

MANIFEST = "data/manifests/e3_comparison_v1.json"


def write_once(path: Path, value):
    body = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
    if path.exists() and path.read_bytes() != body:
        raise ValueError("Immutable comparison material differs: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(body)


def anchor(case, question_index: int, transition: str | None = None):
    script = case["script"]
    index = question_index - 1
    language = script.initial_language if index < 0 else script.stages[index].reply_language
    previous = script.initial_language if index <= 0 else script.stages[index - 1].reply_language
    if transition and transition != f"{previous}_to_{language}":
        raise ValueError("Declared switch does not precede actual question")
    return {
        "after_stage": index,
        "expected_language": language,
        "transition": transition,
        "message_prefix": case["public"].first_user_message
        if index < 0
        else script.stages[index].reply,
        "question_stage_label": script.stages[question_index].label,
    }


def inserted_return_roundtrip(case, proposal_index: int, language: str):
    """已送达订单：用户临时改取消，被拒后重述退货但尚未选择商品，制造自然追问。"""
    raw = case["script"].model_dump(mode="json")
    stages = list(raw["stages"])
    current = case["script"].initial_intent
    for stage in case["script"].stages[:proposal_index]:
        current = apply_update(current, stage.update)
    original = stages[proposal_index]
    after = apply_update(current, FactUpdate.model_validate(original["update"]))
    if current.tool_name != "return_delivered_order_items" or after.tool_name != current.tool_name:
        raise ValueError("Roundtrip overlay requires a delivered-return proposal")
    oid = str(after.arguments["order_id"])
    opposite = "en" if language == "zh" else "zh"
    cancel = {
        "tool_name": "cancel_pending_order",
        "arguments": {"order_id": oid, "reason": "no longer needed"},
    }
    original["reply_language"] = opposite
    original["reply"] = (
        f"Change my request: cancel order {oid}; reason no longer needed. "
        "I have not formally approved any proposal."
        if opposite == "en"
        else f"现在改为取消订单{oid}，原因是不再需要（no longer needed）；之前提议尚未正式确认。"
    )
    original["update"] = cancel
    partial = {k: v for k, v in after.arguments.items() if k != "item_ids"}
    method = str(partial["payment_method_id"])
    refusal = {
        "trigger": "refusal",
        "label": "overlay-cancel-refused-back-to-return",
        "reply_language": language,
        "reply": (
            f"Understood. Return order {oid} instead, refund method {method}. "
            "I have not decided the return item IDs; please ask me."
            if language == "en"
            else f"明白，改回订单{oid}退货，退款方式{method}。具体退哪些商品尚未决定，请先问我。"
        ),
        "update": {"tool_name": "return_delivered_order_items", "arguments": partial},
    }
    item_ids = after.arguments["item_ids"]
    if not isinstance(item_ids, list):
        raise ValueError("Expected item list")
    ids = ", ".join(str(v) for v in item_ids)
    question = {
        "trigger": "question",
        "slot": "items",
        "label": "overlay-return-items-after-language-switch",
        "reply_language": language,
        "reply": f"Return only item IDs {ids}; retain the stated refund method."
        if language == "en"
        else f"只退商品编号{ids}，退款方式沿用刚才指定的。",
        "update": {"arguments": {"item_ids": after.arguments["item_ids"]}},
    }
    stages[proposal_index : proposal_index + 1] = [original, refusal, question]
    raw["stages"] = stages
    script = UserScript.model_validate(raw)
    if script.final_intent != case["script"].final_intent:
        raise ValueError("Overlay changed final business goal")
    case["script"] = script
    case["overlay_notes"] = (
        "Added explicit cancel/refusal/return-item clarification after an existing proposal; "
        "same source identity and final goal, separate version, not language-only generalization."
    )
    return proposal_index + 2


def overlay_switch(case, split: str, position: int):
    if split == "validation" and position == 10:
        raw = case["script"].model_dump(mode="json")
        original = raw["stages"][0]
        reason = original["update"]["arguments"].pop("reason")
        oid = original["update"]["arguments"]["order_id"]
        original["reply"] = (
            f"Do not approve the prior proposal. Cancel order {oid} instead. "
            "I have not provided the cancellation reason; ask me."
        )
        raw["stages"].insert(
            1,
            {
                "trigger": "question",
                "slot": "reason",
                "label": "overlay-cancel-reason-after-switch",
                "reply_language": "en",
                "reply": f"The cancellation reason is {reason}.",
                "update": {"arguments": {"reason": reason}},
            },
        )
        case["script"] = UserScript.model_validate(raw)
        case["overlay_notes"] = (
            "Withheld new cancellation reason at intent switch; then disclosed on question"
        )
        return 1, "zh_to_en"
    if split == "validation":
        return inserted_return_roundtrip(case, 0, "zh"), "en_to_zh"
    if position == 10:
        return inserted_return_roundtrip(case, 2, "en"), "zh_to_en"
    return 3, "en_to_zh"


def build(root: Path):
    groups, files, source_refs = {}, {}, {}
    original_dev, dev_manifest = load_dev(root)
    diagnostics = diagnostic_cases(root)
    sources = {"development": deepcopy(original_dev) + diagnostics}
    source_refs.update(dev_manifest["files"])
    for split in ("validation", "test"):
        values, old = load_holdout(root, split)
        sources[split] = deepcopy(values)
        source_refs.update(old["files"])
    signatures = []
    for split, values in sources.items():
        names = []
        for position, case in enumerate(values):
            original_goal = case["script"].final_intent.model_copy(deep=True)
            case["anchor"] = None
            if split == "development":
                if position >= 12:
                    i = position - 12
                    case["anchor"] = anchor(
                        case,
                        1 if i < 4 else 0,
                        ("en_to_zh" if i == 2 else "zh_to_en" if i == 3 else None),
                    )
            elif (
                position in (1, 3, 8, 9, 10, 11)
                if split == "validation"
                else position in (0, 1, 8, 9, 10, 11)
            ):
                if position >= 10:
                    q, transition = overlay_switch(case, split, position)
                else:
                    q = next(
                        i for i, s in enumerate(case["script"].stages) if s.trigger == "question"
                    )
                    transition = None
                case["anchor"] = anchor(case, q, transition)
            if split != "development" or position < 12:
                signatures.append(signature(case["script"]))
            assert case["script"].final_intent == original_goal
            rel = f"data/tasks/private/e3_comparison_v1/{split}/{case['public'].task_id}.json"
            payload = {
                "public": case["public"].model_dump(mode="json"),
                "script": case["script"].model_dump(mode="json"),
                "criteria": case["criteria"].model_dump(mode="json"),
                "anchor": case["anchor"],
                "overlay_notes": case.get("overlay_notes", "unchanged source"),
            }
            write_once(root / rel, payload)
            files[rel] = sha(root / rel)
            names.append(rel)
        anchors = [c["anchor"] for c in values if c["anchor"]]
        counts = {
            lang: sum(a["expected_language"] == lang for a in anchors) for lang in ("zh", "en")
        }
        assert len(anchors) == 6 and counts == {"zh": 3, "en": 3}, (split, counts)
        assert sum(a["transition"] == "zh_to_en" for a in anchors) == 1
        assert sum(a["transition"] == "en_to_zh" for a in anchors) == 1
        groups[split] = {
            "tasks": len(values),
            "repetitions": 2,
            "files": names,
            "primary_anchors": 6,
            "expected_primary_per_group": 12,
        }
    assert len(signatures) == len(set(signatures)) == 36
    manifest = {
        "suite_id": "e3-matched-skill-comparison-v1",
        "prepared_on": "2026-10-28",
        "protocol_sha256": sha(root / "configs/evaluation/e3_protocol_v2_1.json"),
        "rules_sha256": sha(root / "configs/evaluation/e3_skill_selection_v1.json"),
        "source_files": source_refs,
        "files": files,
        "groups": groups,
        "structural_signature_count": 36,
        "review": (
            "Assistant author review; source identities/final goals preserved; "
            "no real holdout execution"
        ),
        "limitations": [
            "Shared retail mechanisms and correlated repetitions",
            "Language overlays add workflow phases",
            "Diagnostics reuse seen facts; not an independent generalization benchmark",
        ],
    }
    write_once(root / MANIFEST, manifest)
    return manifest


def load(root: Path, split: str):
    manifest = json.loads((root / MANIFEST).read_text("utf-8"))
    if split not in manifest["groups"]:
        raise ValueError("Unknown comparison split")
    for label, name in (
        ("rules_sha256", "configs/evaluation/e3_skill_selection_v1.json"),
        ("protocol_sha256", "configs/evaluation/e3_protocol_v2_1.json"),
    ):
        if sha(root / name) != manifest[label]:
            raise ValueError("Frozen protocol/rules mismatch")
    for mapping in ("files", "source_files"):
        if any(sha(root / p) != digest for p, digest in manifest[mapping].items()):
            raise ValueError("Comparison/source file changed")
    cases = []
    for name in manifest["groups"][split]["files"]:
        value = json.loads((root / name).read_text("utf-8"))
        cases.append(
            {
                "public": VisibleTaskInput.model_validate(value["public"]),
                "script": UserScript.model_validate(value["script"]),
                "criteria": CaseCriteria.model_validate(value["criteria"]),
                "anchor": value["anchor"],
                "overlay_notes": value["overlay_notes"],
            }
        )
    return cases, manifest


def planned_pairs(root: Path, split: str):
    cases, _ = load(root, split)
    return [(case["public"].task_id, repeat) for case in cases for repeat in (1, 2)]
