"""真实Runtime/SQLite + 显式脚本模型；只测驱动，不测模型能力。"""

import asyncio
from pathlib import Path

import pytest

from aftersalesagent.evaluation.nl_replay import ScriptedNLReplay
from aftersalesagent.evaluation.nl_runner import run_nl_case
from aftersalesagent.evaluation.nl_suite import load
from aftersalesagent.policies.application import load_application_policy
from aftersalesagent.schemas.contracts import RunContext, RunLimits

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("index", [4, 5, 6, 7, 8, 9])
def test_nl_multi_turn_fixture_drives_real_confirmation_without_secret_or_future_input(
    index, retail_template, tmp_path
):
    case = load(ROOT, "dev")[0][index]
    seen = []

    class SpyReplay(ScriptedNLReplay):
        async def generate(self, request):
            seen.append(request)
            return await super().generate(request)

    async def execute():
        return await run_nl_case(
            case,
            template=retail_template,
            folder=tmp_path / "episode",
            backend_factory=lambda: SpyReplay(case.plan),
            policy=load_application_policy(ROOT),
            context=RunContext(
                environment_id="later", limits=RunLimits(max_tokens=48000, max_turn_tokens=24000)
            ),
        )

    row = asyncio.run(execute())
    assert row["verification"]["success"], row
    assert row["usage"]["unknown_usage_calls"] > 0
    first_users = [m.content for m in seen[0].messages if m.role == "user"]
    assert first_users == [case.public.first_user_message]
    if case.plan.correction or case.plan.plain_yes:
        assert row["stale_probe"] == {"status": "rejected", "reason": "confirmation_stale"}
    if case.plan.missing_slot:
        assert row["clarification_answered"]


def test_model_receives_authoritative_card_invalidation_state(retail_template, tmp_path):
    import json

    case = load(ROOT, "dev")[0][9]
    seen = []

    class SpyReplay(ScriptedNLReplay):
        async def generate(self, request):
            seen.append(request)
            return await super().generate(request)

    folder = tmp_path / "state-notification"

    async def execute():
        return await run_nl_case(
            case,
            template=retail_template,
            folder=folder,
            backend_factory=lambda: SpyReplay(case.plan),
            policy=load_application_policy(ROOT),
            context=RunContext(
                environment_id="later", limits=RunLimits(max_tokens=48000, max_turn_tokens=24000)
            ),
        )

    row = asyncio.run(execute())
    assert row["verification"]["success"]
    conversation = json.loads((folder / "conversation.json").read_text("utf-8"))
    original_id = conversation[0]["pending_action"]["action_id"]
    requests = [
        r
        for r in seen
        if [m.content for m in r.messages if m.role == "user"][-1] == case.plan.plain_yes
    ]
    assert requests
    notifications = [
        json.loads(m.content.removeprefix("Application state update: "))
        for r in requests
        for m in r.messages
        if m.role == "system" and m.content.startswith("Application state update: ")
    ]
    assert any(
        n["superseded_action_id"] == original_id and n["active_pending_action"] is None
        for n in notifications
    )
