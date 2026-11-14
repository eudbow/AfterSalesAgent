"""只用于工程验收的脚本模型，允许读取用户计划；不是LLM基线或语言理解测试。"""

import json
import re

from aftersalesagent.evaluation.nl_suite import UserPlan
from aftersalesagent.schemas.contracts import ModelResponse, ToolCall


class ScriptedNLReplay:
    def __init__(self, plan: UserPlan):
        self.plan = plan
        self.calls = 0

    async def generate(self, request):
        self.calls += 1
        users = [m.content for m in request.messages if m.role == "user"]
        if self.plan.missing_slot and len(users) == 1:
            return ModelResponse(
                content="请告诉我取消原因？"
                if self.plan.missing_slot == "reason"
                else "请提供新地址的邮编？"
            )
        results = []
        for m in request.messages:
            if m.role == "tool":
                result = json.loads(m.content)
                if result.get("status") == "ok":
                    results.append(result.get("data") or {})

        def call(name, args):
            return ModelResponse(
                tool_calls=[ToolCall(call_id=f"nl-replay-{self.calls}", name=name, arguments=args)]
            )

        if not any(r.get("user_id") == self.plan.source_user_id for r in results):
            email = re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]+", users[0])
            assert email
            return call("find_user_id_by_email", {"email": email.group()})
        read = next(
            (r for r in results if r.get("order_id") == self.plan.order_id and "status" in r), None
        )
        if read is None:
            return call("get_order_details", {"order_id": self.plan.order_id})
        if self.plan.tool_name is None:
            return ModelResponse(content=f"订单 {self.plan.order_id} 状态pending，待处理。")
        if self.plan.tool_name == "cancel_pending_order" and read["status"] == "delivered":
            return ModelResponse(content=f"订单 {self.plan.order_id} 已送达，不能取消。")
        changed = (self.plan.correction and len(users) > 1) or self.plan.missing_slot
        args = self.plan.final_arguments if changed else self.plan.initial_arguments
        return call(self.plan.tool_name, args)
