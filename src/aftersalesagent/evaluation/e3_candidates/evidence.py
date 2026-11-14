"""只从允许的实际开发trace投影公开观察；绝不打开私有脚本、criteria或全库快照。"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from aftersalesagent.business.service import ARGUMENTS
from aftersalesagent.evaluation.e3_v2.review import save

SENSITIVE = {
    "email",
    "first_name",
    "last_name",
    "name",
    "address1",
    "address2",
    "city",
    "state",
    "country",
    "zip",
    "phone",
    "phone_number",
    "user_id",
    "order_id",
    "item_id",
    "product_id",
    "payment_method_id",
    "action_id",
    "tracking_id",
}
PUBLIC_RESULT_KEYS = {"status", "data", "error", "code", "message", "retryable"}
PUBLIC_CALL_KEYS = {"name", "arguments", "tool_name"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_local(root: Path, path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to((root / ".local").resolve()):
        raise ValueError("Output/input must stay under project .local")
    return resolved


class Redactor:
    """跨同一包稳定替换实体；模式匹配不是完美匿名化，联网前还须显式阅读审核。"""

    def __init__(self):
        self.values: dict[str, str] = {}

    def add(self, value):
        if isinstance(value, str) and value and len(value) >= 2:
            self.values.setdefault(value, f"ENTITY_{len(self.values) + 1:04}")

    def collect(self, value, key=""):
        if isinstance(value, dict):
            for field, child in value.items():
                if key in {"payment_methods", "variants"}:
                    self.add(field)
                self.collect(child, field)
        elif isinstance(value, list):
            for child in value:
                self.collect(child, key)
        elif (key in SENSITIVE or key == "item_ids") and value not in ARGUMENTS:
            self.add(value)

    def text(self, text: str) -> str:
        # 邮箱/长数字ID/信用卡或礼品卡ID，即使没有结构化观察也先识别。
        for match in re.findall(
            r"[A-Za-z0-9_.+%-]+@[A-Za-z0-9.-]+\.[A-Za-z]+|"
            r"#[A-Za-z]\d+|\b(?:credit_card|gift_card|paypal)_[A-Za-z0-9]+\b|"
            r"\b\d{5,}\b|\b\d{1,5} [A-Z][A-Za-z ]{1,50}"
            r"(?:Street|Road|Avenue|Lane|Drive|Boulevard)\b",
            text,
        ):
            self.add(match)
        # 单次替换不二次改写ENTITY编号；字母数字边界避免把OR替换进普通单词。
        if not self.values:
            return text
        pattern = (
            r"(?<![A-Za-z0-9_])(?:"
            + "|".join(re.escape(v) for v in sorted(self.values, key=len, reverse=True))
            + r")(?![A-Za-z0-9_])"
        )
        return re.sub(pattern, lambda match: self.values[match.group()], text)

    def apply(self, value) -> Any:
        if isinstance(value, dict):
            return {
                self.text(k) if re.search(r"[0-9@#]", k) else k: self.apply(v)
                for k, v in value.items()
            }
        if isinstance(value, list):
            return [self.apply(v) for v in value]
        return self.text(value) if isinstance(value, str) else value


def project(events: list[dict]) -> list[dict]:
    """最小白名单：实际用户消息、模型动作/回复、业务工具观察；无评分、未来计划、系统全文。"""
    visible, seen_users = [], 0
    for event in events:
        payload, kind = event["payload"], event["kind"]
        if kind == "model_request":
            users = [m for m in payload["messages"] if m["role"] == "user"]
            for message in users[seen_users:]:
                visible.append({"kind": "user_message", "text": message["content"]})
            seen_users = len(users)
        elif kind == "model_response":
            visible.append(
                {
                    "kind": kind,
                    "text": payload.get("content", ""),
                    "tool_calls": [
                        {"name": call["name"], "arguments": call["arguments"]}
                        for call in payload.get("tool_calls", [])
                    ],
                }
            )
        elif kind == "tool_call":
            visible.append(
                {"kind": kind, **{k: v for k, v in payload.items() if k in PUBLIC_CALL_KEYS}}
            )
        elif kind == "confirmation":
            visible.append(
                {
                    "kind": "formal_confirmation",
                    "approved": payload["approved"],
                    "action_id": payload["action_id"],
                }
            )
        elif kind == "tool_result":
            # 轨迹采用result内嵌形状；未知包装拒绝静默扩大白名单。
            observation = payload
            if not {"status", "data", "error"} <= set(observation):
                raise ValueError("Unknown tool-result trace schema")
            visible.append(
                {
                    "kind": kind,
                    "observation": {
                        k: v for k, v in observation.items() if k in PUBLIC_RESULT_KEYS
                    },
                }
            )
    if not visible or not any(item["kind"] == "user_message" for item in visible):
        raise ValueError("No actual public observations")
    return visible


def export_evidence(root: Path, baseline_path: Path, selected: list[str], output: Path):
    output = safe_local(root, output)
    if output.exists() or not 1 <= len(selected) <= 8 or len(selected) != len(set(selected)):
        raise ValueError("New directory and 1..8 unique development traces required")
    baseline = json.loads(baseline_path.read_text("utf-8"))
    allow = {
        name.replace("\\", "/"): digest
        for name, digest in baseline["candidate_trace_inputs"][
            "allowed_development_trace_files"
        ].items()
    }
    redactor, episodes, bindings = Redactor(), [], {}
    for index, name in enumerate(selected, 1):
        name = name.replace("\\", "/")
        path = safe_local(root, root / name)
        if name not in allow or sha(path) != allow[name]:
            raise ValueError("Trace is not in immutable development allowlist")
        events = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
        visible = project(events)
        redactor.collect(visible)
        # 再遍历业务观察/调用内部字段；不读取另外的数据库或用户资料。
        for item in visible:
            redactor.collect(item)
        episodes.append({"evidence_id": f"DEV-{index:02}", "observations": visible})
        bindings[f"DEV-{index:02}"] = {"source": name, "sha256": sha(path)}
    packet = {"format": "e3-public-development-evidence-v1", "episodes": redactor.apply(episodes)}
    output.mkdir(parents=True)
    save(output / "packet.json", packet)
    save(output / "_private_source_map.json", bindings)
    save(
        output / "manifest.json",
        {
            "packet_sha256": sha(output / "packet.json"),
            "baseline_sha256": sha(baseline_path),
            "episodes": len(episodes),
            "requires_explicit_review": True,
            "privacy_limit": "Heuristic pseudonymization; explicit content review required.",
            "excluded": [
                "criteria",
                "future scripts",
                "holdouts",
                "full database snapshots",
                "scores",
                "review group map",
                "system policy from trace",
            ],
        },
    )
    return {"episodes": len(episodes), "packet_sha256": sha(output / "packet.json")}


def review_packet(directory: Path, *, packet_sha256: str, reviewer: str):
    """调用者须实际阅读packet。哈希绑定确认记录，不宣称匿名化可自动证明。"""
    if not reviewer.strip() or sha(directory / "packet.json") != packet_sha256:
        raise ValueError("Explicit reviewer and exact packet digest required")
    receipt = directory / "review_receipt.json"
    if receipt.exists():
        raise ValueError("Do not overwrite evidence review")
    value = {
        "reviewer": reviewer,
        "reviewer_kind": "assistant" if reviewer == "Codex assistant" else "human",
        "packet_sha256": packet_sha256,
        "approved_for_generator": True,
        "meaning": "Explicit packet-content review, not privacy proof",
    }
    save(receipt, value)
    return value


def checked_packet(directory: Path) -> dict:
    packet = json.loads((directory / "packet.json").read_text("utf-8"))
    manifest = json.loads((directory / "manifest.json").read_text("utf-8"))
    receipt = json.loads((directory / "review_receipt.json").read_text("utf-8"))
    if (
        sha(directory / "packet.json") != manifest["packet_sha256"]
        or receipt["packet_sha256"] != manifest["packet_sha256"]
        or not receipt["approved_for_generator"]
    ):
        raise ValueError("Unreviewed or changed evidence packet")
    return packet
