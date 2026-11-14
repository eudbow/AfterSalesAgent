"""额外澄清分支的单列诊断测试，不修改原协议和原评分。"""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from aftersalesagent.evaluation import e3_dev_runner
from aftersalesagent.evaluation.e3_dev_suite import load
from aftersalesagent.evaluation.e3_reason_diagnostic import (
    BASE_USER,
    ReasonAskingReplay,
    ReasonClarifyingUser,
    run_diagnostic,
)
from aftersalesagent.policies.application import load_application_policy
from aftersalesagent.schemas.contracts import MessageEvent, RunContext, RunLimits

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("index", [7, 8, 11])
@pytest.mark.parametrize("group", ["S0", "S1"])
def test_optional_known_reason_branch_without_future_leak(index, group, tmp_path, retail_template):
    case = load(ROOT)[0][index]
    seen = []

    class Spy(ReasonAskingReplay):
        async def generate(self, request):
            seen.append(request.model_copy(deep=True))
            return await super().generate(request)

    async def run():
        return await run_diagnostic(
            case,
            folder=tmp_path / "episode",
            template=retail_template,
            factory=lambda: Spy(case["plan"]),
            policy=load_application_policy(ROOT),
            context=RunContext(
                environment_id="assigned",
                limits=RunLimits.model_validate_json(
                    (ROOT / "configs/m3_live_limits.json").read_bytes()
                ),
            ),
            group=group,
            repetition=1,
            root=ROOT,
        )

    row = asyncio.run(run())
    assert row["verification"]["success"], row
    assert any(a["step"] == "allowed_reason_reconfirmation" for a in row["driver_actions"])
    assert [m.content for m in seen[0].messages if m.role == "user"] == [
        case["public"]["first_user_message"]
    ]
    first_content = "\n".join(m.content for m in seen[0].messages)
    assert "取消原因确认：" not in first_content
    for step in case["plan"]["steps"]:
        assert step["reply"] not in first_content
    assert e3_dev_runner.WorkflowUser is BASE_USER


def test_optional_branch_is_bounded_and_not_a_general_answer_oracle():
    case = load(ROOT)[0][7]
    driver = ReasonClarifyingUser(case["plan"])
    result = SimpleNamespace(
        status="completed",
        reply="Could you confirm the cancellation reason?",
        session=SimpleNamespace(pending_action=None),
    )
    first = driver.next_event(result)
    assert isinstance(first, MessageEvent)
    assert first.content == "取消原因确认：我已经不需要这个订单了（no longer needed）。"
    assert driver.index == 0  # 不把额外问题当作执行了未来意图切换。
    assert driver.next_event(result) is None
    assert driver.failures == ["expected_proposal_missing:cancel_to_address"]


def test_user_provider_restored_even_if_diagnostic_raises(monkeypatch):
    async def fail(*args, **kwargs):
        raise RuntimeError("offline-injected")

    monkeypatch.setattr(e3_dev_runner, "run_case", fail)
    with pytest.raises(RuntimeError, match="offline-injected"):
        asyncio.run(run_diagnostic())
    assert e3_dev_runner.WorkflowUser is BASE_USER
