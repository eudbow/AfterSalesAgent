"""把已见开发任务另存为v2，补充英语和切换语言对照；不构造新留出集。"""

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path

from aftersalesagent.business.service import ARGUMENTS
from aftersalesagent.evaluation.e3_dev_suite import load as load_v1
from aftersalesagent.evaluation.e3_v2.contracts import (
    SLOT_FIELD,
    FactUpdate,
    Intent,
    Language,
    Stage,
    UserScript,
    apply_update,
)
from aftersalesagent.evaluation.suite import CaseCriteria
from aftersalesagent.schemas.contracts import VisibleTaskInput

PROTOCOL = "configs/evaluation/e3_protocol_v2.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_protocol(root: Path):
    protocol = json.loads((root / PROTOCOL).read_text("utf-8"))
    if protocol["version"] != "2.0.0":
        raise ValueError("Unexpected formal user protocol version")
    return protocol


def next_target(plan, index):
    if index < len(plan["steps"]):
        step = plan["steps"][index]
        if step["trigger"] == "proposal":
            return Intent(tool_name=step["expected_tool"], arguments=step["expected_arguments"])
        if step["trigger"] == "question":
            later = next(
                (s for s in plan["steps"][index + 1 :] if s["trigger"] == "proposal"), None
            )
            target = (
                {
                    "tool_name": later["expected_tool"],
                    "arguments": deepcopy(later["expected_arguments"]),
                }
                if later
                else deepcopy(plan["final"])
            )
            target["arguments"].pop(SLOT_FIELD[step["slot"]], None)
            return Intent.model_validate(target)
    return Intent.model_validate(plan["final"])


def build(root: Path):
    source_cases, source_manifest = load_v1(root)
    public, plans, goals = [], [], []
    review = []
    for index, source in enumerate(source_cases, 1):
        original = source["plan"]
        task_id = f"e3v2-dev-{index:02}"
        message = source["public"]["first_user_message"]
        email = re.search(r"[A-Za-z0-9_.+%-]+@[A-Za-z0-9.-]+\.[A-Za-z]+", message)
        assert email
        if original["steps"][0]["trigger"] == "refusal":
            initial = Intent(
                tool_name="cancel_pending_order",
                arguments={"order_id": original["order_id"], "reason": "no longer needed"},
            )
        else:
            initial = next_target(original, 0)
        state = initial
        language: Language = "en" if index in {1, 10, 11} else "zh"
        replies = [step["reply"] for step in original["steps"]]
        reply_languages: list[Language] = [language] * len(replies)
        if index == 1:
            args = initial.arguments
            message = (
                f"Hello, my email is {email.group()}. Please change order {original['order_id']} "
                f"to {args['address1']}, address line 2 Unit 8, Seattle, WA, USA, zip 98101."
            )
            replies[0] = (
                "Please correct the street to 850 Birch Road and clear address line 2. "
                "Keep the other fields unchanged."
            )
        if index == 3:
            replies[0] = (
                "Please correct the street to 900 Pine Avenue. Keep the other fields unchanged."
            )
            reply_languages = ["en", "zh"]
        if index == 10:
            final = original["final"]["arguments"]
            message = (
                f"Hello, my email is {email.group()}. "
                f"I no longer want order {original['order_id']}. "
                "Please cancel it even though it is delivered, or explain if policy disallows it."
            )
            replies[0] = (
                f"Understood. Instead, return only item {final['item_ids'][0]} from that order "
                f"and use the original refund method {final['payment_method_id']}."
            )
        if index == 11:
            final = original["final"]["arguments"]
            message = (
                f"Hello, my email is {email.group()}. Return item {final['item_ids'][0]} "
                f"from order {original['order_id']}. I have not decided the refund method; "
                "please ask me before choosing one."
            )
            replies[1] = "Okay, please proceed with that return proposal."
            reply_languages = ["zh", "en"]
        stages = []
        for stage_index, step in enumerate(original["steps"]):
            after = next_target(original, stage_index + 1)
            args = (
                after.arguments
                if after.tool_name != state.tool_name
                else {
                    key: value
                    for key, value in after.arguments.items()
                    if key not in state.arguments or state.arguments[key] != value
                }
            )
            update = FactUpdate(
                tool_name=after.tool_name if after.tool_name != state.tool_name else None,
                arguments=args,
            )
            expected = (
                Intent(tool_name=step["expected_tool"], arguments=step["expected_arguments"])
                if step["trigger"] == "proposal"
                else None
            )
            stages.append(
                Stage(
                    trigger=step["trigger"],
                    label=step["label"],
                    expected=expected,
                    slot=step.get("slot"),
                    reply=replies[stage_index],
                    reply_language=reply_languages[stage_index],
                    update=update,
                )
            )
            state = apply_update(state, update)
        plan = UserScript(
            task_id=task_id,
            source_user_id=original["source_user_id"],
            source_task=source["public"]["task_id"],
            family=original["family"],
            title=original["title"],
            initial_language=language,
            known_identity={"email": email.group()},
            initial_intent=initial,
            stages=tuple(stages),
            final_intent=Intent.model_validate(original["final"]),
        )
        # 验证卡片与最终参数，缺项状态本身故意不满足完整工具Schema。
        for stage in plan.stages:
            if stage.expected:
                ARGUMENTS[stage.expected.tool_name].model_validate(stage.expected.arguments)
        ARGUMENTS[plan.final_intent.tool_name].model_validate(plan.final_intent.arguments)
        public.append({"task_id": task_id, "first_user_message": message})
        plans.append(plan.model_dump(mode="json"))
        goal = CaseCriteria.model_validate(source["criteria"]).model_copy(
            update={"task_id": task_id}
        )
        goals.append(goal.model_dump(mode="json"))
        review.append(
            {
                "task_id": task_id,
                "source_task": plan.source_task,
                "initial_language": language,
                "reply_languages": reply_languages,
                "initial_known_fields": sorted(initial.arguments),
                "updates": [stage.update.model_dump(mode="json") for stage in stages],
            }
        )

    def lines(rows):
        return "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ).encode()

    contents = {
        "data/tasks/e3_v2_dev/inputs.jsonl": lines(public),
        "data/tasks/private/e3_v2_dev/user_scripts.jsonl": lines(plans),
        "data/tasks/private/e3_v2_dev/criteria.jsonl": lines(goals),
    }
    manifest = {
        "suite_id": "e3-formal-protocol-v2-dev",
        "protocol_id": load_protocol(root)["protocol_id"],
        "protocol_file": PROTOCOL,
        "protocol_sha256": sha(root / PROTOCOL),
        "source_suite_id": source_manifest["suite_id"],
        "source_task_files": source_manifest["files"],
        "tasks": 12,
        "split": "development_only",
        "language_groups": {
            "zh_only": 8,
            "en_only": 2,
            "language_switch": 2,
        },
        "files": {rel: hashlib.sha256(body).hexdigest() for rel, body in contents.items()},
        "review": review,
        "limitations": [
            "Reuses seen v1 development users and scenarios; not fresh held-out data",
            (
                "Assistant-curated progressive fact annotations; "
                "structural checks do not prove semantics"
            ),
            "No validation/test split or model-effectiveness result in this delivery",
            "Language heuristics require explicit review; no automatic final quality claim",
        ],
    }
    contents["data/manifests/e3_v2_dev_suite.json"] = (
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    ).encode()
    for rel, body in contents.items():
        path = root / rel
        if path.exists() and path.read_bytes() != body:
            raise ValueError("V2 frozen assets differ; use a new suite version")
    for rel, body in contents.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(body)
    return manifest


def load(root: Path):
    manifest = json.loads((root / "data/manifests/e3_v2_dev_suite.json").read_text("utf-8"))
    if sha(root / PROTOCOL) != manifest["protocol_sha256"]:
        raise ValueError("Protocol rubric hash mismatch")
    tables = {}
    for rel, digest in {**manifest["source_task_files"], **manifest["files"]}.items():
        path = (root / rel).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path) != digest:
            raise ValueError("V2 source/task hash mismatch")
        if rel in manifest["files"]:
            rows = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
            table = {row["task_id"]: row for row in rows}
            if len(table) != len(rows):
                raise ValueError("Duplicate task IDs")
            tables[path.name] = table
    inputs, scripts, criteria = (
        tables[k] for k in ("inputs.jsonl", "user_scripts.jsonl", "criteria.jsonl")
    )
    if not (inputs.keys() == scripts.keys() == criteria.keys()) or len(inputs) != 12:
        raise ValueError("V2 task IDs/count mismatch")
    protocol = load_protocol(root)
    cases = []
    for task in inputs:
        public = VisibleTaskInput.model_validate(inputs[task])
        script = UserScript.model_validate(scripts[task])
        goal = CaseCriteria.model_validate(criteria[task])
        if script.max_events != protocol["simulator"]["maximum_events"]:
            raise ValueError("Script event budget does not match the protocol")
        if script.final_intent.arguments != goal.arguments or script.source_user_id != goal.user_id:
            raise ValueError("User intent/criteria mismatch")
        if script.known_identity["email"] not in public.first_user_message:
            raise ValueError("Initial email is not actually disclosed")
        cases.append({"public": public, "script": script, "criteria": goal})
    return cases, manifest
