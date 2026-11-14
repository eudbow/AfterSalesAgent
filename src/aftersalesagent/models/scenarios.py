"""离线交互演示的固定脚本；按工具观察推进，不推断任意自然语言意图。"""

from __future__ import annotations

import json
from typing import Literal

from aftersalesagent.schemas.contracts import ModelRequest, ModelResponse, ToolCall

ScenarioName = Literal["query", "cancel", "address", "return"]

SCENARIOS: dict[str, dict] = {
    "query": {
        "label": "查询订单",
        "prompt": "我是 Yara Muller，邮编85041，请查询订单 #W5056519。",
        "steps": [
            (
                "find_user_id_by_name_zip",
                {"first_name": "Yara", "last_name": "Muller", "zip": "85041"},
            ),
            ("get_order_details", {"order_id": "#W5056519"}),
        ],
    },
    "cancel": {
        "label": "取消订单",
        "prompt": "我是 Yara Muller，邮编85041。请取消 #W5056519，原因是下错单。",
        "steps": [
            (
                "find_user_id_by_name_zip",
                {"first_name": "Yara", "last_name": "Muller", "zip": "85041"},
            ),
            ("get_order_details", {"order_id": "#W5056519"}),
            ("cancel_pending_order", {"order_id": "#W5056519", "reason": "ordered by mistake"}),
        ],
    },
    "address": {
        "label": "修改收货地址",
        "prompt": "我是 Yara Muller，邮编85041。请把 #W5056519 的地址改为演示中的新地址。",
        "steps": [
            (
                "find_user_id_by_name_zip",
                {"first_name": "Yara", "last_name": "Muller", "zip": "85041"},
            ),
            ("get_order_details", {"order_id": "#W5056519"}),
            (
                "modify_pending_order_address",
                {
                    "order_id": "#W5056519",
                    "address1": "123 Demo Street",
                    "address2": "",
                    "city": "Seattle",
                    "state": "WA",
                    "country": "USA",
                    "zip": "98101",
                },
            ),
        ],
    },
    "return": {
        "label": "提交退货申请",
        "prompt": "我的邮箱是 mia.garcia2723@example.com，请退回 #W5490111 中的演示键盘。",
        "steps": [
            ("find_user_id_by_email", {"email": "mia.garcia2723@example.com"}),
            ("get_order_details", {"order_id": "#W5490111"}),
            (
                "return_delivered_order_items",
                {
                    "order_id": "#W5490111",
                    "item_ids": ["1421289881"],
                    "payment_method_id": "credit_card_3124723",
                },
            ),
        ],
    },
}


class ScenarioMockBackend:
    def __init__(self, scenario: str):
        if scenario not in SCENARIOS:
            raise ValueError("Unknown offline scenario")
        self.scenario = scenario

    async def generate(self, request: ModelRequest) -> ModelResponse:
        start = max((i for i, m in enumerate(request.messages) if m.role == "user"), default=-1)
        observed = [m for m in request.messages[start + 1 :] if m.role == "tool"]
        if observed:
            result = json.loads(observed[-1].content)
            if result["status"] == "error":
                return ModelResponse(
                    content="演示流程被业务规则拒绝：" + result["error"]["message"]
                )
        steps = SCENARIOS[self.scenario]["steps"]
        if len(observed) < len(steps):
            name, args = steps[len(observed)]
            return ModelResponse(
                tool_calls=[
                    ToolCall(
                        call_id=f"demo-{len(observed)}",
                        name=name,
                        arguments=args,
                    )
                ]
            )
        result = json.loads(observed[-1].content)
        data = result.get("data") or {}
        return ModelResponse(
            content=(
                f"已查询订单 {data.get('order_id', '')}，"
                f"当前状态：{data.get('status', '请查看订单面板')}。"
            )
        )
