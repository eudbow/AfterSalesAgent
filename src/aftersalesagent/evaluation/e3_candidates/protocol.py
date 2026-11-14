"""v2.1只增加有界准备许可；继承v2，正式确认和事实披露规则保持不变。"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import ValidationError

from aftersalesagent.business.service import ARGUMENTS
from aftersalesagent.evaluation.e3_v2.user import FormalUser
from aftersalesagent.schemas.contracts import TurnResult

# 只接受明确的“准备提议”问句，不把execute/approve/退款当作准备许可。
PATTERNS = {
    "return_delivered_order_items": (
        r"(?:Shall I|May I|Would you like me to) prepare (?:this|the) return "
        r"(?:request|proposal|confirmation card)\?",
        r"(?:是否|可以)(?:为你|为您)?准备(?:这份|这个|该)?退货(?:提议|申请提议|确认卡片)[？?]",
    ),
    "cancel_pending_order": (
        r"(?:Shall I|May I|Would you like me to) prepare (?:this|the) cancellation "
        r"(?:request|proposal|confirmation card)\?",
        r"(?:是否|可以)(?:为你|为您)?准备(?:这份|这个|该)?取消(?:提议|确认卡片)[？?]",
    ),
    "modify_pending_order_address": (
        r"(?:Shall I|May I|Would you like me to) prepare (?:this|the) address change "
        r"(?:request|proposal|confirmation card)\?",
        r"(?:是否|可以)(?:为你|为您)?准备(?:这份|这个|该)?改址(?:提议|确认卡片)[？?]",
    ),
}


def load_protocol(root: Path) -> dict:
    value = json.loads((root / "configs/evaluation/e3_protocol_v2_1.json").read_text("utf-8"))
    if value["version"] != "2.1.0":
        raise ValueError("Unexpected preparation-permission protocol")
    return value


class PreparationUser(FormalUser):
    """不读future/final来判断是否回答，只检查当前公开意图和工具参数完整性。"""

    def __init__(self, script, *, options=None):
        super().__init__(script, options=options)
        self.preparation_permissions = 0

    def describe(self):
        return super().describe() | {"preparation_permission_maximum": 1, "version": "2.1.0"}

    def next_event(self, result: TurnResult):
        if (
            self.outcome == "running"
            and result.status == "completed"
            and result.session.pending_action is None
        ):
            # 前面可能是事实复述，只对最后一句精确匹配；不接受别的复合问题。
            lines = [line.strip() for line in result.reply.strip().splitlines() if line.strip()]
            question = lines[-1] if lines else ""
            patterns = PATTERNS[self.known.tool_name]
            if any(re.fullmatch(pattern, question, re.I) for pattern in patterns):
                preceding = "\n".join(lines[:-1])
                if re.search(
                    r"[?？]|\b(?:execute|approve|commit|refund now)\b|立即退款|批准执行",
                    preceding,
                    re.I,
                ):
                    return self._stop("coverage_gap", "ambiguous_preparation_permission")
                try:
                    ARGUMENTS[self.known.tool_name].model_validate(self.known.arguments)
                except ValidationError:
                    return self._stop("coverage_gap", "preparation_with_missing_current_fact")
                if (
                    self.preparation_permissions >= 1
                    or self.optional_count >= self.options.total_optional_replies
                    or self.stage_counts[self.index] >= self.options.per_stage
                ):
                    return self._stop("interaction_limit", "repeated_preparation_permission")
                self.preparation_permissions += 1
                self.optional_count += 1
                self.stage_counts[self.index] += 1
                text = (
                    "请仅按我当前已提供的需求准备待确认卡片；这不代表正式批准执行。"
                    if self.language == "zh"
                    else "Prepare a confirmation card only for my current stated request. "
                    "This is not formal approval to execute it."
                )
                return self._message(text, "optional_preparation_permission")
        return super().next_event(result)
