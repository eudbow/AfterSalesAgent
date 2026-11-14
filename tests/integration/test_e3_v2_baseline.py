"""留出隔离、真实入口门槛和复杂阶段工程验证；默认不发HTTP请求。"""

import asyncio
import json
from pathlib import Path

import pytest

from aftersalesagent.config import Settings
from aftersalesagent.evaluation.e3_v2 import baseline
from aftersalesagent.evaluation.e3_v2.holdout import build, load, signature
from aftersalesagent.evaluation.e3_v2.replay import ProtocolReplay
from aftersalesagent.evaluation.e3_v2.runner import run_case
from aftersalesagent.evaluation.e3_v2.suite import load as load_dev
from aftersalesagent.evaluation.e3_v2.suite import load_protocol
from aftersalesagent.policies.application import load_application_policy
from aftersalesagent.schemas.contracts import RunContext, RunLimits

ROOT = Path(__file__).resolve().parents[2]


def test_all_36_cases_have_distinct_users_and_combination_signatures():
    dev = load_dev(ROOT)[0]
    validation, manifest = load(ROOT, "validation")
    test = load(ROOT, "test")[0]
    assert build(ROOT) == manifest
    combined = dev + validation + test
    assert len(combined) == 36
    assert len({c["script"].source_user_id for c in combined}) == 36
    assert len({signature(c["script"]) for c in combined}) == 36
    for split in (validation, test):
        assert len(split) == 12
        languages = [
            "switch"
            if len({c["script"].initial_language, *[s.reply_language for s in c["script"].stages]})
            > 1
            else c["script"].initial_language
            for c in split
        ]
        assert (
            languages.count("zh") == 8 and languages.count("en") == languages.count("switch") == 2
        )
        for case in split:
            assert case["script"].final_intent.arguments == case["criteria"].arguments
            assert case["script"].source_user_id == case["criteria"].user_id
            assert case["script"].source_user_id not in manifest["excluded_users"]


def test_signature_does_not_use_language_ids_or_wording():
    case = load(ROOT, "validation")[0][0]
    script = case["script"]
    changed = script.model_copy(
        update={
            "task_id": "new-id",
            "family": "renamed",
            "title": "other wording",
            "initial_language": "en" if script.initial_language == "zh" else "zh",
            "stages": tuple(
                s.model_copy(update={"reply": "Different wording."}) for s in script.stages
            ),
        }
    )
    assert signature(changed) == signature(script)


def test_freeze_hashes_holdout_without_running_its_cases(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Do not execute/load live held-out scripts in a development baseline")

    monkeypatch.setattr(baseline, "load_holdout", forbidden)
    config = baseline.freeze_config(ROOT, Settings(backend="deepseek", model_name="deepseek-flash"))
    assert config["selected_split"] == "dev" and config["episodes"] == 48
    assert config["repetitions"] == 2 and config["maximum_model_requests"] == 576
    assert config["sealed_holdout_files"]
    assert config["groups"]["S0"]["skill_version"] == "none"
    assert config["groups"]["S1"]["skill_version"] == "1.0.0"


@pytest.mark.parametrize("phase", ["validation", "test", "all"])
def test_live_entry_rejects_any_unstaged_or_holdout_phase(tmp_path, phase):
    with pytest.raises(ValueError, match="development"):
        asyncio.run(
            baseline.live(
                tmp_path,
                directory=tmp_path / "not-created",
                phase=phase,
                settings=Settings(),
                secret=None,
                config={},
            )
        )
    assert not (tmp_path / "not-created").exists()


def smoke_fixture(tmp_path, *, fault=None):
    for group in ("S0", "S1"):
        folder = tmp_path / f"e3v2-dev-01_r1_{group}"
        folder.mkdir()
        row = {
            "task_id": "e3v2-dev-01",
            "group": group,
            "scoring": {
                "task_status": "failure",
                "business": {
                    "checks": {
                        "read_expected_order": True,
                        "located_expected_user": True,
                    }
                },
            },
            "usage": {"unknown_usage_calls": 1 if fault == "usage" else 0, "model_calls": 1},
            "terminal_status": "completed",
        }
        (folder / "result.json").write_text(json.dumps(row), encoding="utf-8")
        response = {
            "kind": "model_response",
            "payload": {
                "provider_metadata": {"id": "duplicate" if fault == "duplicate" else group},
                "input_tokens": None if fault == "usage" else 10,
                "output_tokens": 2,
            },
        }
        events = [response]
        if fault == "error":
            events.append({"kind": "model_error", "payload": {"code": "model_protocol_error"}})
        (folder / "trace.jsonl").write_text(
            "".join(json.dumps(e) + "\n" for e in events), encoding="utf-8"
        )


@pytest.mark.parametrize("fault", [None, "usage", "duplicate", "error"])
def test_smoke_checks_usage_not_semantic_score_and_rejects_provider_errors(tmp_path, fault):
    smoke_fixture(tmp_path, fault=fault)
    gate = baseline.smoke_gate(tmp_path)
    assert gate["ready"] == (fault is None)
    assert gate["task_status"] == ["failure", "failure"]


@pytest.mark.parametrize(("split", "index"), [("validation", 3), ("test", 11)])
def test_complex_holdout_driver_never_sends_future_user_steps(
    split, index, tmp_path, retail_template
):
    case = load(ROOT, split)[0][index]
    seen = []

    class Spy(ProtocolReplay):
        async def generate(self, request):
            seen.append(request.model_copy(deep=True))
            return await super().generate(request)

    async def execute():
        return await run_case(
            case,
            folder=tmp_path / "episode",
            template=retail_template,
            backend_factory=lambda: Spy(case["script"], variant="clarify"),
            policy_text=load_application_policy(ROOT),
            context=RunContext(
                environment_id="assigned",
                limits=RunLimits.model_validate_json(
                    (ROOT / "configs/m3_live_limits.json").read_bytes()
                ),
            ),
            protocol=load_protocol(ROOT),
            group="S0",
            transport="native",
        )

    row = asyncio.run(execute())
    assert row["scoring"]["task_status"] == "success", row["scoring"]
    first = "\n".join(m.content for m in seen[0].messages)
    assert [m.content for m in seen[0].messages if m.role == "user"] == [
        case["public"].first_user_message
    ]
    assert all(s.reply not in first for s in case["script"].stages)
