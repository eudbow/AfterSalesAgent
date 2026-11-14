"""单列用户协议诊断：接纳一次正常的取消原因复核，不修改 Agent 或原实验。"""

from aftersalesagent.evaluation import e3_dev_runner as original
from aftersalesagent.schemas.contracts import MessageEvent

BASE_USER = original.WorkflowUser


class ReasonClarifyingUser(BASE_USER):
    def __init__(self, plan):
        super().__init__(plan)
        self.reason_confirmed_at = set()

    def next_event(self, result):
        if (
            not self.finished
            and result.status not in {"failed", "truncated", "rejected", "handoff"}
            and result.session.pending_action is None
            and original.asks(result.reply, "reason")
            and self.index not in self.reason_confirmed_at
        ):
            target = (
                self.plan["steps"][self.index] if self.index < len(self.plan["steps"]) else None
            )
            tool = target.get("expected_tool") if target else self.plan["final"]["tool_name"]
            arguments = (
                target.get("expected_arguments") if target else self.plan["final"]["arguments"]
            )
            if tool == "cancel_pending_order" and arguments:
                # 只能重述用户当前已表达的意图，不能把未来步骤或评分答案拿来补参数。
                reason = arguments.get("reason")
                if reason in {"no longer needed", "ordered by mistake"}:
                    self.reason_confirmed_at.add(self.index)
                    text = {
                        "no longer needed": (
                            "取消原因确认：我已经不需要这个订单了（no longer needed）。"
                        ),
                        "ordered by mistake": "取消原因确认：是我下错单了（ordered by mistake）。",
                    }[reason]
                    self.actions.append({"step": "allowed_reason_reconfirmation", "message": text})
                    return MessageEvent(
                        event_id=f"reason-reconfirmation-{self.index}", content=text
                    )
        return super().next_event(result)


class ReasonAskingReplay(original.WorkflowReplay):
    """离线模型固定发出一次原因复核，仅验证新增用户分支。"""

    async def generate(self, request):
        # 额外原因复核不推进原始脚本的阶段。此适配仅用于可见声明的离线脚本模型。
        filtered = [
            m
            for m in request.messages
            if not (m.role == "user" and m.content.startswith("取消原因确认："))
        ]
        response = await super().generate(request.model_copy(update={"messages": filtered}))
        if response.tool_calls and response.tool_calls[0].name == "cancel_pending_order":
            last_user = next(m for m in reversed(request.messages) if m.role == "user")
            if not last_user.content.startswith("取消原因确认："):
                from aftersalesagent.schemas.contracts import ModelResponse

                return ModelResponse(content="请再次确认取消原因？")
        return response


async def run_diagnostic(*args, **kwargs):
    # 只在独立、顺序运行的评测进程临时替换用户模拟器；finally 恢复。
    # 不改 Runtime、业务代码、模型输入生成逻辑、工具、预算或 Skill。
    original.WorkflowUser = ReasonClarifyingUser
    try:
        return await original.run_case(*args, **kwargs)
    finally:
        original.WorkflowUser = BASE_USER
