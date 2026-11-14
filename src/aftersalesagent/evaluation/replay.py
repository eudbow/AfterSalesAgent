"""离线工程回放，仅读取客户公开REQUEST_DATA；不读取私有判分目标。"""

import json

from aftersalesagent.evaluation.suite import TOOL
from aftersalesagent.schemas.contracts import ModelRequest, ModelResponse, ToolCall


class PublicRequestReplay:
    async def generate(self, request: ModelRequest) -> ModelResponse:
        index = next(
            i
            for i, m in enumerate(request.messages)
            if m.role == "user" and "\nREQUEST_DATA:\n" in m.content
        )
        data = json.loads(request.messages[index].content.split("\nREQUEST_DATA:\n", 1)[1])
        observed = [m for m in request.messages[index + 1 :] if m.role == "tool"]
        plan = [
            ("find_user_id_by_email", {"email": data["email"]}),
            ("get_order_details", {"order_id": data["order_id"]}),
        ]
        if data["intent"] != "query":
            plan.append((TOOL[data["intent"]], data["arguments"]))
        if len(observed) < len(plan):
            name, args = plan[len(observed)]
            return ModelResponse(
                tool_calls=[ToolCall(call_id=f"replay-{len(observed)}", name=name, arguments=args)]
            )
        result = json.loads(observed[-1].content)
        if result["status"] == "error":
            return ModelResponse(content=f"Cannot cancel delivered order {data['order_id']}.")
        order = result["data"]
        return ModelResponse(content=f"Order {order['order_id']} is {order['status']}.")
