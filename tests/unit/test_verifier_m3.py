"""Verifier必须拒绝伪成功；不通过调用被测业务实现来生成期望。"""

from copy import deepcopy
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from aftersalesagent.evaluation.pricing import PriceSchedule, estimate_cost
from aftersalesagent.evaluation.suite import CaseCriteria, load_cases, public_data
from aftersalesagent.evaluation.verifier import verify
from aftersalesagent.schemas.contracts import TraceEvent


def event(kind, sequence, **payload):
    return TraceEvent(
        event_id=str(sequence),
        session_id="s",
        sequence=sequence,
        kind=kind,
        payload=payload,
        created_at=datetime.now(UTC),
    )


def state():
    return {
        "snapshot": {
            "orders": {"o": {"status": "pending", "payment_history": []}},
            "users": {},
            "products": {},
        },
        "versions": {"o": 0},
        "operations": [],
    }


def criteria(kind="query"):
    return CaseCriteria.model_validate(
        {"task_id": "unit", "kind": kind, "user_id": "u", "order_id": "o", "arguments": {}}
    )


def test_unchanged_state_and_claimed_success_without_tools_is_not_success():
    initial = state()
    verdict = verify(criteria(), initial, deepcopy(initial), [], "Order o pending", "completed")
    assert not verdict.success
    assert verdict.checks["no_unauthorized_actual_changes"]
    assert not verdict.checks["located_expected_user"]


def test_actual_unconfirmed_change_is_separate_from_failure_to_complete_goal():
    initial = state()
    final = deepcopy(initial)
    final["snapshot"]["orders"]["o"]["status"] = "cancelled"
    verdict = verify(criteria(), initial, final, [], "done", "completed")
    assert not verdict.checks["no_unauthorized_actual_changes"]
    final = deepcopy(initial)
    final["versions"]["o"] = 2
    verdict = verify(criteria(), initial, final, [], "Order o pending", "completed")
    assert not verdict.checks["no_duplicate_actual_execution"]


def test_frozen_suite_entity_split_and_public_negative_intent():
    root = Path(__file__).resolve().parents[2]
    all_cases = []
    users = []
    for split in ("dev", "validation", "test"):
        cases, manifest = load_cases(root, split)
        assert len(cases) == 18
        users.append({case.criteria.user_id for case in cases})
        all_cases += cases
    assert manifest["task_count"] == 54
    assert not users[0] & users[1] and not users[0] & users[2] and not users[1] & users[2]
    for case in all_cases:
        data = public_data(case.public)
        assert "criteria" not in case.public.model_dump()
        if case.criteria.kind == "refuse_cancel":
            assert data["intent"] == "cancel"
        if case.criteria.kind == "decline":
            assert not case.approve_matching


def test_cost_requires_complete_usage_cache_and_explicit_prices():
    pricing = PriceSchedule(
        input_hit_per_million=Decimal("0.1"),
        input_miss_per_million=Decimal("1"),
        output_per_million=Decimal("2"),
        source_url="https://example.invalid/test-only",
        verified_on="test",
        rate_label="test",
    )
    events = [
        event("model_request", 0),
        event(
            "model_response",
            1,
            input_tokens=100,
            output_tokens=10,
            provider_metadata={"prompt_cache_hit_tokens": 20},
        ),
    ]
    cost = estimate_cost(events, pricing)
    assert cost is not None
    assert cost["amount"] == "0.000102"
    assert estimate_cost(events, None) is None
    assert estimate_cost(events + [event("model_request", 2)], pricing) is None
    assert (
        estimate_cost(
            [
                event("model_request", 0),
                event("model_response", 1, input_tokens=None, output_tokens=None),
            ],
            pricing,
        )
        is None
    )
