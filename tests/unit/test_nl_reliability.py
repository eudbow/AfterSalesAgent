"""NL任务/用户脚本边界与伪成功反例；默认无真实HTTP。"""

from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from aftersalesagent.evaluation.nl_driver import ScriptedUser, asks_slot
from aftersalesagent.evaluation.nl_runner import reply_evidence
from aftersalesagent.evaluation.nl_suite import CHALLENGES, build, load
from aftersalesagent.schemas.contracts import (
    ConfirmationEvent,
    MessageEvent,
    PendingAction,
    SessionView,
    TurnResult,
)

ROOT = Path(__file__).resolve().parents[2]


def turn(action=None, reply=""):
    return TurnResult(
        session=SessionView(session_id="s", environment_id="e", pending_action=action),
        status="waiting_confirmation" if action else "completed",
        reply=reply,
    )


def proposal(plan, *, initial=False):
    return PendingAction(
        action_id="a",
        tool_name=plan.tool_name,
        arguments=deepcopy(plan.initial_arguments if initial else plan.final_arguments),
        order_id=plan.order_id,
        order_version="0",
        summary="proposal",
        expires_at=datetime.now(UTC) + timedelta(minutes=15),
    )


def test_suite_is_frozen_36_disjoint_entities_and_no_request_data():
    files, manifest = build(ROOT)
    users, orders, families = [], [], []
    for split in ("dev", "validation", "test"):
        cases, loaded = load(ROOT, split)
        assert loaded == manifest and len(cases) == 12
        assert all(sum(c.plan.challenge == key for c in cases) == 2 for key in CHALLENGES)
        users.append({c.criteria.user_id for c in cases})
        orders.append({c.criteria.order_id for c in cases})
        families.append({c.plan.family for c in cases})
        for c in cases:
            assert "REQUEST_DATA" not in c.public.first_user_message
            assert c.plan.source_user_id == c.criteria.user_id
            assert c.plan.final_arguments == c.criteria.arguments
            assert c.plan.order_id == c.criteria.order_id
            assert not set(c.public.model_dump()) & {"plan", "criteria", "approve"}
    for groups in (users, orders, families):
        assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
    for rel, payload in files.items():
        assert (ROOT / rel).read_bytes() == payload


@pytest.mark.parametrize("index,slot", [(4, "reason"), (5, "zip")])
def test_missing_field_is_only_released_after_explicit_question(index, slot):
    case = load(ROOT, "dev")[0][index]
    driver = ScriptedUser(case.plan)
    assert not driver.answered
    other = "请告诉我邮箱？" if slot == "reason" else "请说明取消原因？"
    assert driver.next_event(turn(reply=other)) is None
    assert not driver.answered and driver.failures == ["missing_required_question"]
    driver = ScriptedUser(case.plan)
    reply = "请说明取消原因？" if slot == "reason" else "Please provide the postal code?"
    event = driver.next_event(turn(reply=reply))
    assert isinstance(event, MessageEvent)
    assert event.content == case.plan.answer
    assert driver.answered


def test_guessed_missing_field_proposal_is_rejected_even_when_correct():
    case = load(ROOT, "dev")[0][5]
    driver = ScriptedUser(case.plan)
    event = driver.next_event(turn(proposal(case.plan)))
    assert isinstance(event, ConfirmationEvent)
    assert event.approved is False
    assert "proposal_before_required_clarification" in driver.failures


def test_wrong_proposal_is_not_silently_fixed_or_approved():
    case = load(ROOT, "dev")[0][3]
    driver = ScriptedUser(case.plan)
    action = proposal(case.plan).model_copy(update={"order_id": "#OTHER"})
    event = driver.next_event(turn(action))
    assert isinstance(event, ConfirmationEvent)
    assert not event.approved and driver.failures == ["unexpected_proposal"]


def test_correction_is_a_message_not_an_approval_and_changes_expected_proposal():
    case = load(ROOT, "dev")[0][6]
    driver = ScriptedUser(case.plan)
    event = driver.next_event(turn(proposal(case.plan, initial=True)))
    assert isinstance(event, MessageEvent)
    assert event.kind == "message" and event.content == case.plan.correction
    assert driver.corrected and not driver.finished
    event = driver.next_event(turn(proposal(case.plan)))
    assert isinstance(event, ConfirmationEvent)
    assert event.kind == "confirmation" and event.approved


def test_plain_yes_and_decline_never_implicitly_approve():
    cases = load(ROOT, "dev")[0]
    driver = ScriptedUser(cases[9].plan)
    event = driver.next_event(turn(proposal(cases[9].plan)))
    assert isinstance(event, MessageEvent)
    assert event.kind == "message" and driver.said_yes and not driver.finished
    driver = ScriptedUser(cases[8].plan)
    event = driver.next_event(turn(proposal(cases[8].plan)))
    assert isinstance(event, ConfirmationEvent)
    assert event.kind == "confirmation" and not event.approved


def test_question_and_reply_rules_do_not_accept_unrelated_claims():
    assert not asks_slot("邮箱是多少？", "zip")
    case = load(ROOT, "dev")[0][0]
    assert not reply_evidence(case.criteria, "订单已经取消")
    assert reply_evidence(case.criteria, f"{case.criteria.order_id} 尚未发货")
