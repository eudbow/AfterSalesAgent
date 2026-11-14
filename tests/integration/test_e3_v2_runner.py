"""真实 Runtime/SQLite 的v2工程验收；脚本模型不计真实模型能力。"""

import asyncio
from pathlib import Path

import pytest

from aftersalesagent.evaluation import e3_dev_runner
from aftersalesagent.evaluation.e3_v2.contracts import SimulatorOptions, UserFactory
from aftersalesagent.evaluation.e3_v2.replay import ProtocolReplay
from aftersalesagent.evaluation.e3_v2.runner import run_case
from aftersalesagent.evaluation.e3_v2.scoring import aggregate
from aftersalesagent.evaluation.e3_v2.suite import load, load_protocol
from aftersalesagent.evaluation.e3_v2.user import FormalUser
from aftersalesagent.policies.application import load_application_policy
from aftersalesagent.schemas.contracts import RunContext, RunLimits

ROOT = Path(__file__).resolve().parents[2]


async def execute(
    case,
    tmp_path,
    template,
    *,
    variant="clarify",
    transport="native",
    user_factory: UserFactory = FormalUser,
    backend_factory=None,
):
    return await run_case(
        case,
        folder=tmp_path / "episode",
        template=template,
        backend_factory=backend_factory
        or (lambda: ProtocolReplay(case["script"], variant=variant)),
        policy_text=load_application_policy(ROOT),
        context=RunContext(
            environment_id="assigned",
            limits=RunLimits.model_validate_json(
                (ROOT / "configs/m3_live_limits.json").read_bytes()
            ),
        ),
        protocol=load_protocol(ROOT),
        group="S0",
        transport=transport,
        user_factory=user_factory,
    )


@pytest.mark.parametrize("index", range(12))
def test_all_v2_scripts_run_without_private_future_input(index, tmp_path, retail_template):
    case = load(ROOT)[0][index]
    seen = []

    class Spy(ProtocolReplay):
        async def generate(self, request):
            seen.append(request.model_copy(deep=True))
            return await super().generate(request)

    row = asyncio.run(
        execute(case, tmp_path, retail_template, backend_factory=lambda: Spy(case["script"]))
    )
    assert row["scoring"]["task_status"] == "success", row["scoring"]
    assert row["scoring"]["business"]["success"] and row["scoring"]["safety"]["passed"]
    first = "\n".join(message.content for message in seen[0].messages)
    assert [m.content for m in seen[0].messages if m.role == "user"] == [
        case["public"].first_user_message
    ]
    for stage in case["script"].stages:
        assert stage.reply not in first
    for key in ("initial_intent", "known_identity", "source_user_id", "expected_arguments"):
        assert key not in first
    assert row["scoring"]["language"]["status"] == "pending_review"
    assert all(
        record["heuristic_assessment"] == "matches"
        for record in row["scoring"]["language"]["records"]
    )
    if index in {2, 10}:
        assert {r["expected_language"] for r in row["scoring"]["language"]["records"]} == {
            "zh",
            "en",
        }


@pytest.mark.parametrize("index", [7, 11])
def test_actual_mcp_still_uses_formal_confirmation(index, tmp_path, retail_template):
    case = load(ROOT)[0][index]
    row = asyncio.run(execute(case, tmp_path, retail_template, transport="mcp"))
    assert row["scoring"]["task_status"] == "success"
    assert row["scoring"]["safety"]["passed"]
    assert row["scoring"]["interaction"]["optional_replies"] > 0
    assert (tmp_path / "episode/mcp.jsonl").is_file()


@pytest.mark.parametrize("index", [0, 1, 2, 10])
def test_wrong_language_does_not_rewrite_business_score(index, tmp_path, retail_template):
    case = load(ROOT)[0][index]
    row = asyncio.run(execute(case, tmp_path, retail_template, variant="wrong_language"))
    assert row["scoring"]["business"]["success"]
    assert row["scoring"]["task_status"] == "success"
    assert all(
        r["heuristic_assessment"] == "mismatch" for r in row["scoring"]["language"]["records"]
    )
    assert aggregate([row])["S0"]["reviewed_language_pass_rate"] is None


def test_unknown_user_branch_is_review_not_automatic_model_failure(tmp_path, retail_template):
    case = load(ROOT)[0][7]
    row = asyncio.run(execute(case, tmp_path, retail_template, variant="coverage_gap"))
    assert row["scoring"]["task_status"] == "needs_review"
    assert not row["scoring"]["business"]["success"]
    assert row["scoring"]["safety"]["passed"]
    result = aggregate([row])["S0"]
    assert result["episodes"] == result["needs_review"] == 1
    assert result["failure"] == 0


def test_repeat_question_stops_with_finite_budget(tmp_path, retail_template):
    case = load(ROOT)[0][7]
    row = asyncio.run(execute(case, tmp_path, retail_template, variant="repeat"))
    assert row["scoring"]["task_status"] == "failure"
    assert row["scoring"]["interaction"]["outcome"] == "interaction_limit"
    assert row["scoring"]["interaction"]["optional_replies"] == 1
    assert row["usage"]["model_calls"] < 12


def test_explicit_factories_are_isolated_and_do_not_use_global_v1_user(
    tmp_path, retail_template, monkeypatch
):
    case = load(ROOT)[0][7]
    created = []

    def forbidden(*args, **kwargs):
        raise AssertionError("The old global user must not be used")

    monkeypatch.setattr(e3_dev_runner, "WorkflowUser", forbidden)

    def normal(script):
        result = FormalUser(script)
        created.append(result)
        return result

    def bounded(script):
        result = FormalUser(script, options=SimulatorOptions(total_optional_replies=0))
        created.append(result)
        return result

    async def run():
        return await asyncio.gather(
            execute(case, tmp_path / "normal", retail_template, user_factory=normal),
            execute(case, tmp_path / "bounded", retail_template, user_factory=bounded),
        )

    a, b = asyncio.run(run())
    assert a["scoring"]["task_status"] == "success"
    assert b["scoring"]["interaction"]["outcome"] == "interaction_limit"
    assert len(created) == 2 and created[0] is not created[1]
    assert e3_dev_runner.WorkflowUser is forbidden


@pytest.mark.parametrize("group", ["S0", "S1"])
def test_baseline_label_must_match_actual_skill_selection(group, tmp_path, retail_template):
    case = load(ROOT)[0][0]
    context = RunContext(
        environment_id="assigned",
        skill_id="after-sales-workflow" if group == "S0" else None,
        skill_version="1.0.0" if group == "S0" else "none",
    )
    with pytest.raises(ValueError, match=group):
        asyncio.run(
            run_case(
                case,
                folder=tmp_path / "must_not_exist",
                template=retail_template,
                backend_factory=lambda: ProtocolReplay(case["script"]),
                policy_text=load_application_policy(ROOT),
                context=context,
                protocol=load_protocol(ROOT),
                group=group,
            )
        )
    assert not (tmp_path / "must_not_exist").exists()
