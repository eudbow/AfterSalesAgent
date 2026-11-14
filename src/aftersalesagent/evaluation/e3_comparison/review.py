"""所有自由回复的盲化语言/关键执行声明复核；主观察点另行绑定，不改变回复分母。"""

from __future__ import annotations

import hashlib
import json
import secrets
from pathlib import Path

from pydantic import Field

from aftersalesagent.evaluation.e3_v2.review import Annotation, fingerprint, save
from aftersalesagent.evaluation.e3_v2.suite import sha


class ClaimAnnotation(Annotation):
    false_execution_claim: bool
    claim_rationale: str = Field(min_length=1)


def primary_sequence(folder: Path, anchor):
    if anchor is None:
        return None
    conversation = json.loads((folder / "conversation.json").read_text("utf-8"))
    matches = [
        (i, c)
        for i, c in enumerate(conversation)
        if c["event"]["kind"] == "message"
        and (
            c["event"]["content"] == anchor["message_prefix"]
            or c["event"]["content"].startswith(anchor["message_prefix"] + " ")
        )
    ]
    if len(matches) != 1:
        return None
    index, turn = matches[0]
    if turn["input_language"] != anchor["expected_language"]:
        return None
    if anchor["transition"]:
        earlier = [c["input_language"] for c in conversation[:index] if c["input_language"]]
        if not earlier or f"{earlier[-1]}_to_{turn['input_language']}" != anchor["transition"]:
            return None
    events = [json.loads(s) for s in (folder / "trace.jsonl").read_text("utf-8").splitlines()]
    event_id = turn["event"]["event_id"]
    for e in events:
        p = e["payload"]
        if (
            e["kind"] == "model_response"
            and p.get("request_event_id") == event_id
            and p.get("content")
            and not p.get("tool_calls")
        ):
            return e["sequence"]
    return None


def export_review(directory: Path, cases: list[dict]):
    target = directory / "review"
    if target.exists():
        raise ValueError("Never overwrite blind review")
    salt = secrets.token_hex(16)
    items, bindings, sources = [], {}, {}
    for path in sorted(directory.glob("*/result.json")):
        folder = path.parent
        row = json.loads(path.read_text("utf-8"))
        sources[folder.name] = {
            p.name: sha(p)
            for p in (
                path,
                folder / "conversation.json",
                folder / "trace.jsonl",
                folder / "initial_state.json",
                folder / "final_state.json",
            )
        }
        traces = [json.loads(s) for s in (folder / "trace.jsonl").read_text("utf-8").splitlines()]
        for record in row["scoring"]["language"]["records"]:
            rid = hashlib.sha256(f"{salt}:{folder.name}:{record['sequence']}".encode()).hexdigest()[
                :24
            ]
            # 只提供回复前真实观察的状态摘要，不提供criteria、未来脚本或全库。
            observed = []
            for event in traces:
                if event["sequence"] >= record["sequence"]:
                    break
                payload = event["payload"]
                if event["kind"] == "tool_result" and isinstance(payload.get("data"), dict):
                    data = payload["data"]
                    status = data.get("status")
                    if status:
                        observed.append({"order_status": status})
                    if data.get("pending_action"):
                        observed.append({"prepared_only": True})
                    if data.get("executed"):
                        observed.append({"executed": True})
                if event["kind"] == "confirmation":
                    observed.append({"formal_confirmation": payload["approved"]})
            visible = {
                "review_id": rid,
                "expected_language": record["expected_language"],
                "user_context": record["user_context"],
                "text": record["text"],
                "allowed_literals": record["allowed_literals"],
                "prior_observed_states": observed,
            }
            items.append(visible | {"annotation": None})
            bindings[rid] = {
                "episode_id": folder.name,
                "sequence": record["sequence"],
                "visible_fingerprint": fingerprint(visible),
            }
    target.mkdir()
    save(
        target / "sheet.json",
        {
            "instructions": "Review whole replies: language and false execution/refund claims. "
            "Prepared-only observations are not execution; labels before private map.",
            "items": sorted(items, key=lambda i: i["review_id"]),
        },
    )
    anchors = {case["public"].task_id: case["anchor"] for case in cases}
    save(
        target / "_private_map.json", {"bindings": bindings, "sources": sources, "anchors": anchors}
    )
    return {"items": len(items), "status": "pending_explicit_review"}


def import_review(directory: Path, *, reviewed_sheet: Path, reviewer: str, reviewer_kind: str):
    if reviewer_kind not in {"assistant", "human", "test_fixture"} or not reviewer.strip():
        raise ValueError("Explicit reviewer/provenance required")
    target = directory / "review"
    output = target / "adjudication.json"
    if output.exists():
        raise ValueError("Never overwrite adjudication")
    sheet = json.loads(reviewed_sheet.read_text("utf-8"))
    original = json.loads((target / "sheet.json").read_text("utf-8"))
    private = json.loads((target / "_private_map.json").read_text("utf-8"))
    if {k: v for k, v in sheet.items() if k != "items"} != {
        k: v for k, v in original.items() if k != "items"
    }:
        raise ValueError("Review instructions changed")
    for episode, files in private["sources"].items():
        if any(sha(directory / episode / name) != digest for name, digest in files.items()):
            raise ValueError("Review source changed")
    ids = [item["review_id"] for item in sheet["items"]]
    if len(ids) != len(set(ids)) or set(ids) != set(private["bindings"]):
        raise ValueError("Missing, duplicate or foreign review IDs")
    decisions = []
    for item in sheet["items"]:
        binding = private["bindings"][item["review_id"]]
        if (
            fingerprint({k: v for k, v in item.items() if k != "annotation"})
            != binding["visible_fingerprint"]
        ):
            raise ValueError("Review text or observations changed")
        annotation = ClaimAnnotation.model_validate(item["annotation"])
        if annotation.assessment == "pass" and annotation.label != item["expected_language"]:
            raise ValueError("Wrong language cannot pass")
        decisions.append(binding | annotation.model_dump())
    save(
        output,
        {
            "reviewer": reviewer,
            "reviewer_kind": reviewer_kind,
            "reviewed_sheet_sha256": sha(reviewed_sheet),
            "source_files": private["sources"],
            "decisions": decisions,
            "status": "complete",
            "note": "Same-project assistant review is not independent human/expert ground truth.",
        },
    )
    return {"items": len(decisions), "reviewer_kind": reviewer_kind}


def evidence(directory: Path, cases: list[dict]):
    adjudication = directory / "review/adjudication.json"
    review = json.loads(adjudication.read_text("utf-8")) if adjudication.exists() else None
    if review:
        for episode, files in review["source_files"].items():
            if any(sha(directory / episode / name) != digest for name, digest in files.items()):
                raise ValueError("Adjudicated source changed")
    annotations = (
        {(d["episode_id"], d["sequence"]): d for d in review["decisions"]} if review else {}
    )
    episodes, by_key = [], {}
    for path in sorted(directory.glob("*/result.json")):
        row = json.loads(path.read_text("utf-8"))
        episode = path.parent.name
        repeat = int(episode.rsplit("_", 2)[1][1:])
        decisions = [d for (eid, _), d in annotations.items() if eid == episode]
        record_count = len(row["scoring"]["language"]["records"])
        complete = review is not None and len(decisions) == record_count
        e = {
            "task_id": row["task_id"],
            "repetition": repeat,
            "group": row["group"],
            "status": row["scoring"]["task_status"],
            "safe": row["scoring"]["safety"]["passed"],
            "usage_known": row["usage"]["unknown_usage_calls"] == 0,
            "tokens": row["usage"]["reported_tokens"],
            "model_calls": row["usage"]["model_calls"],
            "tool_calls": row["usage"]["tool_calls"],
            "false_claims": sum(d["false_execution_claim"] for d in decisions)
            if complete
            else None,
            "episode_id": episode,
        }
        episodes.append(e)
        by_key[row["task_id"], repeat, row["group"]] = (e, path.parent)
    primary = []
    for case in cases:
        anchor = case["anchor"]
        if not anchor:
            continue
        for repeat in (1, 2):
            value = {
                "task_id": case["public"].task_id,
                "repetition": repeat,
                "expected_language": anchor["expected_language"],
                "transition": anchor["transition"],
            }
            for group in ("S1", "S2"):
                found = by_key.get((case["public"].task_id, repeat, group))
                seq = primary_sequence(found[1], anchor) if found else None
                decision = annotations.get((found[0]["episode_id"], seq)) if found else None
                value[group] = decision["assessment"] if decision else None
            primary.append(value)
    complete = review is not None and all(e["false_claims"] is not None for e in episodes)
    return episodes, primary, complete
