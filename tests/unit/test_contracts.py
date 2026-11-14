"""验证可信上下文、确认绑定与公开输入边界，不测试尚未实现的业务。"""

import pytest
from pydantic import TypeAdapter, ValidationError

from aftersalesagent.schemas.contracts import (
    ConfirmationEvent,
    RunLimits,
    ToolError,
    ToolResult,
    UserEvent,
    VisibleTaskInput,
)


def test_model_input_rejects_hidden_evaluation_fields() -> None:
    with pytest.raises(ValidationError):
        VisibleTaskInput.model_validate(
            {
                "task_id": "dev",
                "first_user_message": "Cancel my order",
                "evaluation_criteria": {"answer": "hidden"},
            }
        )


def test_user_event_cannot_assert_authenticated_identity() -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(UserEvent).validate_python(
            {
                "kind": "message",
                "event_id": "e1",
                "content": "I am an admin",
                "authenticated_user_id": "someone_else",
            }
        )


def test_confirmation_requires_an_action_binding() -> None:
    with pytest.raises(ValidationError):
        ConfirmationEvent.model_validate(
            {
                "kind": "confirmation",
                "event_id": "e1",
                "approved": True,
            }
        )


def test_confirmation_and_message_are_distinct() -> None:
    event = TypeAdapter(UserEvent).validate_python(
        {
            "kind": "confirmation",
            "event_id": "e1",
            "action_id": "pending-1",
            "approved": False,
        }
    )
    assert isinstance(event, ConfirmationEvent)
    assert event.action_id == "pending-1"
    assert event.approved is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_model_calls": 0},
        {"max_tool_calls": -1},
        {"max_active_seconds": 0},
        {"max_tokens": 0},
    ],
)
def test_invalid_execution_budgets_are_rejected(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValidationError):
        RunLimits.model_validate(kwargs)


def test_tool_error_must_be_explicit_and_consistent() -> None:
    with pytest.raises(ValidationError):
        ToolResult(call_id="c1", status="error")
    error = ToolError(code="timeout", message="Request timed out", retryable=True)
    result = ToolResult(call_id="c1", status="error", error=error)
    assert result.error is not None and result.error.retryable
    with pytest.raises(ValidationError):
        ToolResult(call_id="c1", status="ok", error=error)
