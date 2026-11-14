"""有界用户状态机：仅看用户计划和Agent输出，不访问私有criteria。"""

import re

from aftersalesagent.evaluation.nl_suite import UserPlan
from aftersalesagent.schemas.contracts import ConfirmationEvent, MessageEvent, PendingAction


def asks_slot(reply: str, slot: str) -> bool:
    words = {"reason": r"原因|理由|reason|why", "zip": r"邮编|邮政编码|zip|postal"}
    question = r"[?？]|请|告诉|提供|告知|需要|缺少|could|please|provide|need|what"
    return bool(re.search(words[slot], reply, re.I) and re.search(question, reply, re.I))


def matches(action: PendingAction, plan: UserPlan, *, initial=False) -> bool:
    return (
        action.tool_name == plan.tool_name
        and action.order_id == plan.order_id
        and action.arguments == (plan.initial_arguments if initial else plan.final_arguments)
    )


class ScriptedUser:
    def __init__(self, plan: UserPlan):
        self.plan = plan
        self.messages = 1
        self.answered = False
        self.corrected = False
        self.said_yes = False
        self.finished = False
        self.old_action_id = None
        self.failures: list[str] = []
        self.actions: list[dict] = []

    def _message(self, content: str, kind: str):
        self.messages += 1
        if self.messages > self.plan.max_messages:
            self.failures.append("driver_message_limit")
            self.finished = True
            return None
        self.actions.append({"kind": kind, "message": content})
        return MessageEvent(event_id=f"user-{self.messages}", content=content)

    def next_event(self, result):
        if self.finished:
            return None
        if result.status in {"failed", "truncated", "handoff", "rejected"}:
            self.finished = True
            return None
        action = result.session.pending_action
        if action:
            if self.plan.missing_slot and not self.answered:
                self.failures.append("proposal_before_required_clarification")
                self.finished = True
                return ConfirmationEvent(
                    event_id="reject-premature", action_id=action.action_id, approved=False
                )
            if self.plan.correction and not self.corrected:
                if not matches(action, self.plan, initial=True):
                    self.failures.append("unexpected_initial_proposal")
                    self.finished = True
                    return ConfirmationEvent(
                        event_id="reject-unexpected", action_id=action.action_id, approved=False
                    )
                self.corrected = True
                self.old_action_id = action.action_id
                return self._message(self.plan.correction, "correction")
            if self.plan.plain_yes and not self.said_yes:
                if not matches(action, self.plan):
                    self.failures.append("unexpected_proposal")
                    self.finished = True
                    return ConfirmationEvent(
                        event_id="reject-unexpected", action_id=action.action_id, approved=False
                    )
                self.said_yes = True
                self.old_action_id = action.action_id
                return self._message(self.plan.plain_yes, "plain_yes")
            valid = matches(action, self.plan)
            if not valid:
                self.failures.append("unexpected_proposal")
            self.finished = True
            self.actions.append({"kind": "confirmation", "approved": valid and self.plan.approve})
            return ConfirmationEvent(
                event_id="decision",
                action_id=action.action_id,
                approved=valid and self.plan.approve,
            )
        if self.plan.missing_slot and not self.answered:
            if asks_slot(result.reply, self.plan.missing_slot):
                self.answered = True
                assert self.plan.answer is not None
                return self._message(self.plan.answer, "answer_requested_slot")
            self.failures.append("missing_required_question")
        if self.plan.correction and not self.corrected:
            self.failures.append("initial_proposal_missing")
        if self.plan.plain_yes and not self.said_yes:
            self.failures.append("initial_proposal_missing")
        self.finished = True
        return None
