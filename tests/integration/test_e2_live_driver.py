"""E2真实验收的驱动器回归；只用显式回放模型，不调用API。"""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from aftersalesagent.evaluation.e2_live_runner import E2Replay, E2User, business
from aftersalesagent.evaluation.e2_live_suite import load
from aftersalesagent.evaluation.nl_suite import UserPlan
from aftersalesagent.policies.application import load_application_policy
from aftersalesagent.schemas.contracts import MessageEvent, RunContext, RunLimits

ROOT = Path(__file__).resolve().parents[2]


def test_suite_separates_public_input_and_future_user_script():
    cases, manifest = load(ROOT)
    assert len(cases) == 8 and manifest["episodes"] == 16
    assert sum(c["public"]["kind"] == "knowledge" for c in cases) == 1
    missing = cases[3]
    assert (
        missing["user_script"]["extra"]["item_answer"]
        not in (missing["public"]["first_user_message"])
    )
    assert (
        cases[2]["user_script"]["plan"]["correction"]
        not in (cases[2]["public"]["first_user_message"])
    )


def test_missing_item_driver_does_not_answer_without_question():
    case = load(ROOT)[0][3]
    plan = UserPlan.model_validate(case["user_script"]["plan"])
    extra = case["user_script"]["extra"]
    driver = E2User(plan, extra)
    event = driver.next_event(
        SimpleNamespace(
            session=SimpleNamespace(pending_action=None),
            status="waiting_user",
            reply="请告诉我具体要退哪件商品？",
        )
    )
    assert isinstance(event, MessageEvent) and event.content == extra["item_answer"]
    driver = E2User(plan, extra)
    assert (
        driver.next_event(
            SimpleNamespace(
                session=SimpleNamespace(pending_action=None),
                status="waiting_user",
                reply="请在网页上确认。",
            )
        )
        is None
    )
    assert driver.failures == ["missing_item_question"]


@pytest.mark.parametrize("transport", ["native", "mcp"])
def test_response_loss_replay_uses_real_runtime_and_original_key(
    transport, retail_template, tmp_path
):
    case = load(ROOT)[0][7]
    plan = UserPlan.model_validate(case["user_script"]["plan"])

    async def execute():
        return await business(
            case,
            folder=tmp_path / "episode",
            template=retail_template,
            factory=lambda: E2Replay(plan, case["public"]["scenario"], {}),
            policy=load_application_policy(ROOT),
            context=RunContext(
                environment_id="assigned", limits=RunLimits(max_tokens=48000, max_turn_tokens=24000)
            ),
            transport=transport,
        )

    row = asyncio.run(execute())
    assert row["verification"]["success"], row
    assert row["fault_exercised"] and row["mcp_retry_events"] == 1
    assert row["model_request_events"] == row["model_response_events"] == 3
    assert row["usage"]["unknown_usage_calls"] == 3


def test_return_clarification_is_not_leaked_to_first_model_request(retail_template, tmp_path):
    case = load(ROOT)[0][3]
    plan = UserPlan.model_validate(case["user_script"]["plan"])
    seen = []

    class Spy(E2Replay):
        async def generate(self, request):
            seen.append(request)
            return await super().generate(request)

    async def execute():
        return await business(
            case,
            folder=tmp_path / "episode",
            template=retail_template,
            factory=lambda: Spy(plan, case["public"]["scenario"], case["user_script"]["extra"]),
            policy=load_application_policy(ROOT),
            context=RunContext(
                environment_id="assigned", limits=RunLimits(max_tokens=48000, max_turn_tokens=24000)
            ),
            transport="native",
        )

    row = asyncio.run(execute())
    assert row["verification"]["success"], row
    first = [m.content for m in seen[0].messages if m.role == "user"]
    assert first == [case["public"]["first_user_message"]]
    assert case["user_script"]["extra"]["item_answer"] not in first
