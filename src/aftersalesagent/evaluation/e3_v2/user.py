"""有界、支持合法复核的用户状态机；仅当前已公开事实可生成可选回应。"""

from __future__ import annotations

import re
from collections import Counter
from copy import deepcopy

from aftersalesagent.evaluation.e3_v2.contracts import (
    ADDRESS_FIELDS,
    SLOT_FIELD,
    Intent,
    SimulatorOptions,
    SimulatorState,
    UserScript,
    apply_update,
    normalise,
)
from aftersalesagent.schemas.contracts import ConfirmationEvent, MessageEvent, TurnResult

INQUIRY = re.compile(
    r"[?？]|请.{0,18}(?:提供|告诉|告知|确认|选择|说明)|"
    r"\b(?:could|would|can) you\b|\bplease\b|\bwhich\b|\bwhat\b|"
    r"\bneed (?:to know|you to|the|your)\b",
    re.I,
)
SLOT_WORDS = {
    "reason": r"原因|理由|\breason\b|\bwhy\b",
    "zip": r"邮编|邮政编码|\bzip\b|\bpostal",
    "items": r"哪.{0,6}[件些个]|商品|\bitems?\b|\bitem_?ids?\b|\bproducts?\b",
    "payment": r"退款.*方式|退.*到|支付方式|\bpayment\b|\brefund.{0,20}method|\bhow.{0,35}refund",
    "email": r"邮箱|电子邮件|\be-?mail\b",
    "order_id": r"订单.{0,5}(?:号|编号)|\border (?:id|number)\b",
}
REFUSAL = re.compile(r"cannot|can't|unable|not (?:allowed|possible)|无法|不能|不支持|不允许", re.I)


def requested_slots(reply: str) -> tuple[set[str], bool]:
    # 只看真正提问/请求的句段，避免前面商品列表污染后面的退款方式问题。
    segments = re.split(r"(?<=[?!？。！])\s*|\n+", reply)
    questions = [segment for segment in segments if INQUIRY.search(segment)]
    focus = "\n".join(questions)
    slots = {name for name, pattern in SLOT_WORDS.items() if re.search(pattern, focus, re.I)}
    if (
        re.search(
            r"(?:完整|全部).{0,5}地址|请.{0,8}(?:提供|确认).{0,4}地址|"
            r"(?:provide|confirm|tell me) (?:the |your |full |new |shipping )*address\b",
            focus,
            re.I,
        )
        and "zip" not in slots
    ):
        slots.add("address")
    if re.search(r"which one", focus, re.I) and all(
        reason in reply.lower() for reason in ("no longer needed", "ordered by mistake")
    ):
        slots.add("reason")
    return slots, bool(questions)


def field_values(intent: Intent, identity: dict, slots: set[str]) -> dict | None:
    values = {}
    for slot in sorted(slots):
        if slot == "address":
            if not all(name in intent.arguments for name in ADDRESS_FIELDS):
                return None
            values[slot] = {name: intent.arguments[name] for name in ADDRESS_FIELDS}
        else:
            field = SLOT_FIELD[slot]
            source = identity if slot == "email" else intent.arguments
            if field not in source:
                return None
            values[slot] = deepcopy(source[field])
    return values


def render_answer(values: dict, language: str) -> str:
    labels = {
        "zh": {
            "reason": "取消原因确认",
            "zip": "邮编确认",
            "items": "退货商品编号",
            "payment": "退款方式",
            "email": "邮箱",
            "order_id": "订单号",
            "address": "完整地址",
        },
        "en": {
            "reason": "Cancellation reason",
            "zip": "Zip code",
            "items": "Return item IDs",
            "payment": "Refund method",
            "email": "Email",
            "order_id": "Order ID",
            "address": "Full address",
        },
    }
    parts = []
    for slot, value in values.items():
        if isinstance(value, dict):
            value = ", ".join(f"{key}={item}" for key, item in value.items())
        elif isinstance(value, list):
            value = ", ".join(str(v) for v in value)
        parts.append(f"{labels[language][slot]}: {value}")
    return ("；" if language == "zh" else "; ").join(parts) + ("。" if language == "zh" else ".")


class FormalUser:
    def __init__(self, script: UserScript, *, options: SimulatorOptions | None = None):
        self.script = script
        self.options = options or SimulatorOptions()
        self.index = 0
        self.known = script.initial_intent.model_copy(deep=True)
        self.language = script.initial_language
        self.outcome = "running"
        self.code = None
        self.actions = []
        self.issues = []
        self.old_cards = set()
        self.optional_count = 0
        self.counts = Counter()
        self.stage_counts = Counter()
        self.event_number = 0

    def describe(self) -> dict:
        return {
            "implementation": type(self).__module__ + "." + type(self).__qualname__,
            "options": self.options.model_dump(mode="json"),
        }

    def state(self) -> SimulatorState:
        return SimulatorState.model_validate(
            {
                "step_index": self.index,
                "outcome": self.outcome,
                "code": self.code,
                "language": self.language,
                "known_intent": self.known.model_dump(mode="json"),
                "actions": deepcopy(self.actions),
                "issues": deepcopy(self.issues),
                "optional_replies": self.optional_count,
            }
        )

    def _stop(self, outcome, code, result=None):
        self.outcome, self.code = outcome, code
        self.issues.append({"category": outcome, "code": code, "stage": self.index})
        if result is not None and result.session.pending_action is not None:
            return ConfirmationEvent(
                event_id=self._id(),
                action_id=result.session.pending_action.action_id,
                approved=False,
            )
        return None

    def _id(self):
        self.event_number += 1
        return f"v2-user-event-{self.event_number}"

    def _message(self, text, kind, *, used_fields=()):
        self.actions.append(
            {
                "kind": kind,
                "stage": self.index,
                "message": text,
                "used_fact_fields": sorted(used_fields),
                "language": self.language,
            }
        )
        return MessageEvent(event_id=self._id(), content=text)

    def on_limit(self) -> None:
        self._stop("interaction_limit", "total_event_limit")

    def next_event(self, result: TurnResult):
        if self.outcome != "running":
            return None
        if result.status in {"failed", "truncated"}:
            category = "interaction_limit" if result.status == "truncated" else "runtime_failure"
            return self._stop(category, result.termination_reason or result.status)
        if result.status in {"handoff", "rejected"}:
            return self._stop("agent_failure", "agent_stopped_without_goal")
        stage = self.script.stages[self.index] if self.index < len(self.script.stages) else None
        action = result.session.pending_action
        if action is not None:
            expected = stage.expected if stage is not None else self.script.final_intent
            if expected is None:
                return self._stop("agent_failure", "proposal_before_required_interaction", result)
            if action.tool_name != expected.tool_name or normalise(action.arguments) != normalise(
                expected.arguments
            ):
                return self._stop("agent_failure", "unexpected_proposal_arguments", result)
            if action.action_id in self.old_cards:
                return self._stop("agent_failure", "stale_card_reused", result)
            if stage is None:
                self.outcome = "completed"
                self.actions.append({"kind": "formal_confirmation", "approved": True})
                return ConfirmationEvent(
                    event_id=self._id(), action_id=action.action_id, approved=True
                )
            self.old_cards.add(action.action_id)
            return self._advance_stage(stage)

        slots, inquiry = requested_slots(result.reply)
        if stage is not None and stage.trigger == "question" and stage.slot in slots:
            # 计划中的缺项现在才被询问；其他要求也必须能从本轮披露后的事实回应。
            after = apply_update(self.known, stage.update)
            extras = slots - {stage.slot}
            values = field_values(after, self.script.known_identity, extras)
            if values is None:
                return self._stop("coverage_gap", "unanswerable_combined_question")
            text = stage.reply
            if values:
                text += " " + render_answer(values, stage.reply_language)
            return self._advance_stage(stage, text)
        if stage is not None and stage.trigger == "refusal" and REFUSAL.search(result.reply):
            if re.search(r"取消|\bcancel(?:led|lation)?\b", result.reply, re.I):
                return self._advance_stage(stage)
            return self._stop("coverage_gap", "ambiguous_refusal")
        if inquiry:
            if not slots:
                return self._stop("coverage_gap", "unrecognised_question")
            values = field_values(self.known, self.script.known_identity, slots)
            if values is None:
                return self._stop("coverage_gap", "requested_unrevealed_fact")
            if (
                self.optional_count >= self.options.total_optional_replies
                or self.stage_counts[self.index] >= self.options.per_stage
                or any(
                    self.counts[self.index, slot] >= self.options.per_slot_per_stage
                    for slot in slots
                )
            ):
                return self._stop("interaction_limit", "repeated_or_excessive_clarification")
            self.optional_count += 1
            self.stage_counts[self.index] += 1
            for slot in slots:
                self.counts[self.index, slot] += 1
            return self._message(
                render_answer(values, self.language),
                "optional_known_fact_confirmation",
                used_fields=slots,
            )
        return self._stop("agent_failure", "required_proposal_or_interaction_missing")

    def _advance_stage(self, stage, text=None):
        self.known = apply_update(self.known, stage.update)
        self.language = stage.reply_language
        self.index += 1
        return self._message(text or stage.reply, "scripted_" + stage.trigger)
