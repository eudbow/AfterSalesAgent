"""协议边界、事实公开顺序和语言反例；不调用模型API。"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from aftersalesagent.evaluation.e3_v2.contracts import SimulatorOptions, UserScript
from aftersalesagent.evaluation.e3_v2.language import classify, observed_literals
from aftersalesagent.evaluation.e3_v2.suite import build, load, load_protocol
from aftersalesagent.evaluation.e3_v2.user import FormalUser, requested_slots
from aftersalesagent.schemas.contracts import (
    ConfirmationEvent,
    MessageEvent,
    PendingAction,
    SessionView,
    TurnResult,
)

ROOT = Path(__file__).resolve().parents[2]


def script(index=7):
    return load(ROOT)[0][index]["script"]


def reply(text="", action=None):
    return TurnResult(
        session=SessionView(
            session_id="test-session", environment_id="test", pending_action=action
        ),
        reply=text,
        status="waiting_confirmation" if action else "completed",
    )


def card(intent, action_id="card-1"):
    return PendingAction(
        action_id=action_id,
        tool_name=intent.tool_name,
        arguments=intent.arguments,
        order_id=str(intent.arguments["order_id"]),
        order_version="0",
        summary="fixture",
        expires_at=datetime.now(UTC) + timedelta(minutes=10),
    )


def test_suite_is_separate_version_with_language_controls():
    cases, manifest = load(ROOT)
    assert build(ROOT) == manifest
    assert len(cases) == 12
    assert manifest["language_groups"] == {"zh_only": 8, "en_only": 2, "language_switch": 2}
    assert all(case["public"].task_id.startswith("e3v2-dev-") for case in cases)
    assert set(cases[0]["public"].model_dump()) == {"task_id", "first_user_message"}


def test_future_zip_cannot_be_used_before_intent_switch():
    user = FormalUser(script(11))
    assert "zip" not in user.state().known_intent.arguments
    event = user.next_event(reply("请确认新地址邮编？"))
    assert event is None
    state = user.state()
    assert state.outcome == "coverage_gap" and state.code == "requested_unrevealed_fact"
    assert "98109" not in str(state.actions)


def test_current_known_reason_reconfirmation_then_original_switch_and_zip():
    plan = script(11)
    user = FormalUser(plan)
    event = user.next_event(reply("Could you confirm the cancellation reason and order ID?"))
    assert isinstance(event, MessageEvent) and "no longer needed" in event.content
    assert str(plan.initial_intent.arguments["order_id"]) in event.content
    assert user.state().step_index == 0
    event = user.next_event(reply(action=card(plan.stages[0].expected)))
    assert isinstance(event, MessageEvent)
    assert event.content == plan.stages[0].reply
    assert user.state().step_index == 1
    assert "zip" not in user.state().known_intent.arguments
    event = user.next_event(reply("请告诉我新邮编？"))
    assert isinstance(event, MessageEvent)
    assert event.content == plan.stages[1].reply
    assert user.state().known_intent.arguments["zip"] == "98109"
    event = user.next_event(reply(action=card(plan.final_intent, "card-final")))
    assert isinstance(event, ConfirmationEvent)
    assert event.approved and user.state().outcome == "completed"


def test_repeated_known_question_is_bounded_and_does_not_advance_stage():
    user = FormalUser(script())
    assert user.next_event(reply("Please confirm the cancellation reason?")) is not None
    assert user.state().step_index == 0
    assert user.next_event(reply("Please confirm the cancellation reason?")) is None
    assert user.state().outcome == "interaction_limit"
    assert user.state().optional_replies == 1


def test_optional_global_budget_can_disable_reconfirmation():
    user = FormalUser(script(), options=SimulatorOptions(total_optional_replies=0))
    assert user.next_event(reply("Please confirm the cancellation reason?")) is None
    assert user.state().outcome == "interaction_limit" and user.state().optional_replies == 0


def test_unknown_question_is_coverage_gap_not_agent_error():
    user = FormalUser(script())
    assert user.next_event(reply("Please provide a scanned invoice image?")) is None
    assert user.state().outcome == "coverage_gap"
    assert user.state().issues[0]["category"] == "coverage_gap"


def test_no_question_or_card_is_distinct_planning_failure():
    user = FormalUser(script())
    assert user.next_event(reply("I will take care of that.")) is None
    assert user.state().outcome == "agent_failure"


def test_wrong_card_is_rejected_and_never_approved():
    plan = script()
    wrong = plan.initial_intent.model_copy(
        update={"arguments": plan.initial_intent.arguments | {"reason": "ordered by mistake"}}
    )
    user = FormalUser(plan)
    event = user.next_event(reply(action=card(wrong)))
    assert isinstance(event, ConfirmationEvent) and event.approved is False
    assert user.state().outcome == "agent_failure"


def test_required_slot_cannot_be_guessed_in_early_card():
    plan = script(1)
    user = FormalUser(plan)
    event = user.next_event(reply(action=card(plan.stages[1].expected)))
    assert isinstance(event, ConfirmationEvent)
    assert event.approved is False
    assert user.state().code == "proposal_before_required_interaction"


def test_superseded_card_id_is_not_reused_for_new_goal():
    plan = script()
    user = FormalUser(plan)
    user.next_event(reply(action=card(plan.stages[0].expected, "same-id")))
    event = user.next_event(reply(action=card(plan.final_intent, "same-id")))
    assert isinstance(event, ConfirmationEvent)
    assert event.approved is False and user.state().code == "stale_card_reused"


def test_script_contract_requires_all_final_facts_to_be_revealed():
    data = script().model_dump(mode="json")
    data["final_intent"]["arguments"]["address1"] = "unrevealed secret street"
    with pytest.raises(ValueError, match="fully disclosed"):
        UserScript.model_validate(data)


@pytest.mark.parametrize(
    ("text", "slots"),
    [
        ("Could you confirm the cancellation reason and order ID?", {"reason", "order_id"}),
        ("Your item is a Bookshelf.\nWhich refund method would you like?", {"payment"}),
        ("Please provide item_ids?", {"items"}),
        ("How would you like to receive the refund?", {"payment"}),
        ("请提供完整地址？", {"address"}),
        ("请确认邮箱和订单号？", {"email", "order_id"}),
        ("Allowed reasons: no longer needed, ordered by mistake. Which one applies?", {"reason"}),
    ],
)
def test_question_detection_focuses_on_requested_slots(text, slots):
    actual, inquiry = requested_slots(text)
    assert inquiry and actual == slots


@pytest.mark.parametrize(
    ("text", "expected", "literals", "label", "assessment"),
    [
        ("请提供新的邮政编码。", "zh", set(), "zh", "matches"),
        ("Please provide the new postal code.", "zh", set(), "en", "mismatch"),
        ("Please provide the new postal code. 好", "zh", set(), "mixed", "needs_review"),
        ("请提供新的邮政编码。", "en", set(), "zh", "mismatch"),
        ("Please provide the new postal code.", "en", set(), "en", "matches"),
        ("订单#W123商品Bookshelf已收到，请选择退款方式。", "zh", {"Bookshelf"}, "zh", "matches"),
        ("好", "zh", set(), "undetermined", "needs_review"),
        ("Okay.", "en", set(), "undetermined", "needs_review"),
    ],
)
def test_language_is_provisional_and_not_single_character_detection(
    text, expected, literals, label, assessment
):
    rules = load_protocol(ROOT)["language"]["heuristic"]
    result = classify(text, expected, literals, rules)
    assert result["heuristic_label"] == label
    assert result["heuristic_assessment"] == assessment
    assert result["final_assessment"] is None


def test_literal_whitelist_excludes_arbitrary_explanations():
    result = observed_literals(
        {
            "name": "Bookshelf",
            "options": {"material": "glass"},
            "message": "Please provide all the documents before returning this order.",
            "description": "This long explanatory prose must not be ignored.",
        }
    )
    assert result == {"Bookshelf", "glass"}


def test_ambiguous_refusal_does_not_trigger_future_return_request():
    user = FormalUser(script(9))
    assert user.next_event(reply("I cannot access your account.")) is None
    assert user.state().outcome == "coverage_gap"
    assert user.state().step_index == 0
    assert not user.state().actions


def test_explicit_cancellation_refusal_can_advance_customer_decision():
    plan = script(9)
    user = FormalUser(plan)
    event = user.next_event(reply("This delivered order cannot be cancelled."))
    assert isinstance(event, MessageEvent) and event.content == plan.stages[0].reply
    assert user.state().step_index == 1
