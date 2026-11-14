"""可读用户未来脚本的离线模型；故意覆盖额外提问、语言错误和协议覆盖缺口。"""

from __future__ import annotations

import json

from aftersalesagent.evaluation.e3_v2.contracts import apply_update
from aftersalesagent.evaluation.e3_v2.user import field_values, render_answer
from aftersalesagent.schemas.contracts import ModelResponse, ToolCall

QUESTIONS = {
    "zh": {
        "reason": "请再确认取消原因？",
        "zip": "请确认新地址邮编？",
        "items": "请告诉我具体退哪些商品编号？",
        "payment": "请选择退款方式？",
    },
    "en": {
        "reason": "Please confirm the cancellation reason?",
        "zip": "Please provide the zip code?",
        "items": "Which item IDs would you like to return?",
        "payment": "Which refund method would you like?",
    },
}


class ProtocolReplay:
    def __init__(self, script, *, variant="clarify"):
        if variant not in {"direct", "clarify", "wrong_language", "coverage_gap", "repeat"}:
            raise ValueError("Unknown offline replay variant")
        self.script, self.variant = script, variant
        self.calls = 0

    async def generate(self, request):
        users = [message for message in request.messages if message.role == "user"]
        index, phase_start = 0, 0
        known, language = self.script.initial_intent, self.script.initial_language
        for position, message in enumerate(users[1:], 1):
            if index < len(self.script.stages) and message.content.startswith(
                self.script.stages[index].reply
            ):
                stage = self.script.stages[index]
                known = apply_update(known, stage.update)
                language = stage.reply_language
                index += 1
                phase_start = position
        stage = self.script.stages[index] if index < len(self.script.stages) else None
        response_language = (
            ("en" if language == "zh" else "zh") if self.variant == "wrong_language" else language
        )
        data = [
            json.loads(message.content).get("data") or {}
            for message in request.messages
            if message.role == "tool"
        ]
        if not any(value.get("user_id") == self.script.source_user_id for value in data):
            name, args = "find_user_id_by_email", {"email": self.script.known_identity["email"]}
        elif not any(value.get("order_id") == known.arguments["order_id"] for value in data):
            name, args = "get_order_details", {"order_id": known.arguments["order_id"]}
        else:
            if self.variant == "coverage_gap":
                return ModelResponse(content="Please provide a scanned invoice image?")
            if stage is not None and stage.trigger == "question":
                return ModelResponse(content=QUESTIONS[response_language][stage.slot])
            if stage is not None and stage.trigger == "refusal":
                text = (
                    "已送达订单不能取消。"
                    if response_language == "zh"
                    else ("This delivered order cannot be cancelled.")
                )
                return ModelResponse(content=text)
            if self.variant != "direct":
                slot = {
                    "cancel_pending_order": "reason",
                    "modify_pending_order_address": "zip",
                    "return_delivered_order_items": "payment",
                }[known.tool_name]
                values = field_values(known, self.script.known_identity, {slot})
                if values is not None:
                    answer = render_answer(values, language)
                    already_answered = any(m.content == answer for m in users[phase_start:])
                    if self.variant == "repeat" or not already_answered:
                        return ModelResponse(content=QUESTIONS[response_language][slot])
            target = stage.expected if stage is not None else self.script.final_intent
            assert target is not None
            name, args = target.tool_name, target.arguments
        self.calls += 1
        return ModelResponse(
            tool_calls=[
                ToolCall.model_validate(
                    {"call_id": f"v2-replay-{self.calls}", "name": name, "arguments": args}
                )
            ]
        )
