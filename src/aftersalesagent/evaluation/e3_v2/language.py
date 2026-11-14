"""语言规则只提供候选标签与复核线索；最终结论必须由显式复核给出。"""

from __future__ import annotations

import hashlib
import re

from aftersalesagent.schemas.contracts import TraceEvent

IDENTIFIERS = re.compile(
    r"[A-Za-z0-9_.+%-]+@[A-Za-z0-9.-]+\.[A-Za-z]+|"
    r"#[A-Za-z][0-9]+|\b[a-z]+(?:_[a-z]+)*_[0-9]+\b|"
    r"\b[a-f0-9]{32,64}\b|\b[0-9]+\b",
    re.I,
)
FIELDS = {
    "order_id",
    "item_id",
    "item_ids",
    "product_id",
    "payment_method_id",
    "address1",
    "address2",
    "city",
    "state",
    "country",
    "zip",
    "user_id",
    "action_id",
    "pending",
    "delivered",
    "USA",
    "WA",
    "OR",
    "RMB",
    "USD",
}


def digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def observed_literals(value, *, path=()) -> set[str]:
    """只取结构化实体，绝不把message/description等整段英文当可忽略内容。"""
    found = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"message", "description", "summary", "content"}:
                continue
            found |= observed_literals(child, path=(*path, key))
    elif isinstance(value, list):
        for child in value:
            found |= observed_literals(child, path=path)
    elif isinstance(value, str) and 1 < len(value) <= 80:
        if path and (
            path[-1] in {"name", "product_name", "first_name", "last_name", "email"}
            or "options" in path
            or "address" in path
        ):
            found.add(value)
    return found


def classify(text: str, expected: str, literals: set[str], rules: dict) -> dict:
    residual = IDENTIFIERS.sub(" ", text)
    # 边界匹配，避免把短实体嵌在真实解释单词中的部分也删除。
    for literal in sorted(literals | FIELDS, key=len, reverse=True):
        residual = re.sub(r"(?<!\w)" + re.escape(literal) + r"(?!\w)", " ", residual, flags=re.I)
    han = len(re.findall(r"[\u4e00-\u9fff]", residual))
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", residual)
    if han >= rules["min_han_for_zh"] and len(words) <= rules["max_residual_english_words_for_zh"]:
        label = "zh"
    elif han == 0 and len(words) >= rules["min_words_for_en"]:
        label = "en"
    elif han > 0 and words:
        label = "mixed"
    else:
        label = "undetermined"
    estimate = (
        "matches"
        if label == expected
        else ("mismatch" if label in {"zh", "en"} else "needs_review")
    )
    return {
        "heuristic_label": label,
        "heuristic_assessment": estimate,
        "han_characters": han,
        "residual_english_words": len(words),
        "expected_language": expected,
        "text": text,
        "text_sha256": digest_text(text),
        "allowed_literals": sorted(literals),
        "final_assessment": None,
    }


def language_records(
    events: list[TraceEvent], conversation: list[dict], protocol: dict
) -> list[dict]:
    by_event = {entry["event"]["event_id"]: entry for entry in conversation}
    current = None
    current_user_text = ""
    literals = set()
    records = []
    for event in events:
        if event.kind == "user_message":
            entry = by_event[event.payload["event_id"]]
            current = entry["input_language"]
            current_user_text = entry["event"]["content"]
        if event.kind == "tool_result" and event.payload.get("status") == "ok":
            literals |= observed_literals(event.payload.get("data"))
        if event.kind == "model_response":
            payload = event.payload
            text = str(payload.get("content") or "").strip()
            if text and not payload.get("tool_calls"):
                if current is None:
                    raise ValueError("Model reply has no delivered user-language context")
                record = classify(text, current, literals, protocol["language"]["heuristic"])
                record.update(sequence=event.sequence, user_context=current_user_text)
                records.append(record)
    return records
