"""冻结门槛的正反例；不使用真实API，不从模型猜正确答案。"""

import asyncio
import json
from copy import deepcopy
from pathlib import Path

import pytest

from aftersalesagent.evaluation.e3_comparison.gates import evaluate
from aftersalesagent.evaluation.e3_comparison.materials import load
from aftersalesagent.evaluation.e3_comparison.provider import MonitoredBackend
from aftersalesagent.models.errors import ModelBackendFault
from aftersalesagent.schemas.contracts import ModelMessage, ModelRequest, ModelResponse

ROOT = Path(__file__).resolve().parents[2]
RULES = json.loads((ROOT / "configs/evaluation/e3_skill_selection_v1.json").read_text("utf-8"))


def fixture(split="development"):
    n = 18 if split == "development" else 12
    planned = [(f"case-{i}", r) for i in range(n) for r in (1, 2)]
    episodes = [
        {
            "task_id": task,
            "repetition": repeat,
            "group": g,
            "status": "success",
            "safe": True,
            "usage_known": True,
            "tokens": 1000 if g == "S1" else 1200,
            "model_calls": 5,
            "tool_calls": 5,
            "false_claims": 0,
        }
        for task, repeat in planned
        for g in ("S1", "S2")
    ]
    primary = []
    for i in range(6):
        language = "zh" if i % 2 == 0 else "en"
        transition = "en_to_zh" if i == 4 else "zh_to_en" if i == 5 else None
        for repeat in (1, 2):
            primary.append(
                {
                    "task_id": f"case-{i}",
                    "repetition": repeat,
                    "expected_language": language,
                    "transition": transition,
                    "S1": "fail" if language == "zh" else "pass",
                    "S2": "pass",
                }
            )
    return planned, episodes, primary


def verdict(values, **kwargs):
    planned, episodes, primary = values
    return evaluate(
        RULES, "development", planned, episodes, primary, review_complete=True, **kwargs
    )


@pytest.mark.parametrize("split", ["development", "validation", "test"])
def test_matching_stage_passes_with_actual_language_improvement(split):
    plan, episodes, primary = fixture(split)
    result = evaluate(RULES, split, plan, episodes, primary, review_complete=True)
    assert result["status"] == "passed"
    assert result["metrics"]["primary_language"]["paired_wins"] == 6


@pytest.mark.parametrize(
    "mode",
    [
        "business_regression",
        "business_floor",
        "zh_low",
        "english_regression",
        "switch_regression",
        "no_gain",
        "tokens",
        "model_calls",
        "tool_calls",
    ],
)
def test_rejects_regression_no_gain_and_overhead(mode):
    plan, episodes, primary = fixture()
    s1 = [e for e in episodes if e["group"] == "S1"]
    s2 = [e for e in episodes if e["group"] == "S2"]
    if mode == "business_regression":
        s2[0]["status"] = "failure"
    elif mode == "business_floor":
        for e in s1[:2] + s2[:2]:
            e["status"] = "failure"
    elif mode == "zh_low":
        for p in [p for p in primary if p["expected_language"] == "zh"][:2]:
            p["S2"] = "fail"
    elif mode == "english_regression":
        next(p for p in primary if p["expected_language"] == "en")["S2"] = "fail"
    elif mode == "switch_regression":
        next(p for p in primary if p["transition"] == "en_to_zh")["S2"] = "fail"
    elif mode == "no_gain":
        for p in primary:
            p["S1"] = "pass"
    elif mode == "tokens":
        for e in s2:
            e["tokens"] = 1250
        s2[0]["tokens"] += 1
    else:
        s2[0][mode] += 19  # baseline180，allowed198，actual199：不能取整放宽。
    assert verdict((plan, episodes, primary))["status"] == "rejected"


def test_exact_integer_resource_boundary_is_allowed():
    values = fixture()
    for e in values[1]:
        if e["group"] == "S2":
            e["tokens"] = 1250
    assert verdict(values)["status"] == "passed"


@pytest.mark.parametrize(
    "mode",
    [
        "missing_pair",
        "duplicate_pair",
        "missing_anchor",
        "unreviewed_anchor",
        "needs_review",
        "claims_pending",
        "no_review",
    ],
)
def test_missing_evidence_never_becomes_success(mode):
    plan, episodes, primary = fixture()
    complete = True
    if mode == "missing_pair":
        episodes.pop()
    elif mode == "duplicate_pair":
        episodes[-1] = deepcopy(episodes[0])
    elif mode == "missing_anchor":
        primary.pop()
    elif mode == "unreviewed_anchor":
        primary[0]["S2"] = None
    elif mode == "needs_review":
        episodes[0]["status"] = "needs_review"
    elif mode == "claims_pending":
        episodes[0]["false_claims"] = None
    else:
        complete = False
    result = evaluate(RULES, "development", plan, episodes, primary, review_complete=complete)
    assert result["status"] == "inconclusive_coverage"


@pytest.mark.parametrize(
    "mode", ["provider", "unknown_usage", "S1_unsafe", "S2_unsafe", "false_claim"]
)
def test_technical_and_safety_precedence(mode):
    values = fixture()
    technical = []
    target = next(e for e in values[1] if e["group"] == ("S1" if mode == "S1_unsafe" else "S2"))
    if mode == "provider":
        technical = ["provider_fingerprint_drift"]
    elif mode == "unknown_usage":
        target["usage_known"] = False
    elif mode == "false_claim":
        target["false_claims"] = 1
    else:
        target["safe"] = False
    result = verdict(values, technical=technical)
    assert result["status"] == (
        "reject_and_halt" if mode in {"S2_unsafe", "false_claim"} else "inconclusive_technical"
    )


def test_materials_bind_18_12_12_and_preserve_final_goals():
    counts = {}
    for split in ("development", "validation", "test"):
        cases, manifest = load(ROOT, split)
        counts[split] = len(cases)
        assert sum(c["anchor"] is not None for c in cases) == 6
        assert manifest["structural_signature_count"] == 36
        if split != "development":
            from aftersalesagent.evaluation.e3_v2.holdout import load as original

            source = {c["public"].task_id: c for c in original(ROOT, split)[0]}
            for c in cases:
                assert (
                    c["script"].source_user_id
                    == source[c["public"].task_id]["script"].source_user_id
                )
                assert (
                    c["script"].final_intent == source[c["public"].task_id]["script"].final_intent
                )
                assert c["criteria"] == source[c["public"].task_id]["criteria"]
    assert counts == {"development": 18, "validation": 12, "test": 12}


@pytest.mark.parametrize("mode", ["fingerprint", "model", "duplicate", "usage", "provider_error"])
def test_provider_ledger_preserves_real_observation_before_halt(mode, tmp_path):
    class Backend:
        def __init__(self):
            self.calls = 0

        async def generate(self, request):
            self.calls += 1
            if mode == "provider_error" and self.calls == 2:
                raise ModelBackendFault("model_rate_limited")
            second = self.calls == 2
            return ModelResponse(
                content="fixture",
                input_tokens=None if mode == "usage" and second else 10,
                output_tokens=2,
                finish_reason="stop",
                provider_metadata={
                    "id": "id-1" if mode == "duplicate" and second else f"id-{self.calls}",
                    "model": "different" if mode == "model" and second else "deepseek-flash",
                    "system_fingerprint": "changed"
                    if mode == "fingerprint" and second
                    else "stable",
                },
            )

        async def aclose(self):
            pass

    backend = MonitoredBackend(Backend(), tmp_path / "ledger.jsonl", "episode", "deepseek-flash")
    request = ModelRequest(
        model_name="deepseek-flash",
        tools=[],
        messages=[ModelMessage(role="user", content="fixture")],
    )

    async def run():
        await backend.generate(request)
        with pytest.raises(ModelBackendFault):
            await backend.generate(request)

    asyncio.run(run())
    ledger = [json.loads(s) for s in (tmp_path / "ledger.jsonl").read_text("utf-8").splitlines()]
    assert len(ledger) == 2 and ledger[-1]["error"]
    if mode != "provider_error":
        assert ledger[-1]["response"]["output_tokens"] == 2


def test_foreign_primary_pair_does_not_pass():
    values = fixture()
    values[2][0]["task_id"] = "not-in-this-stage"
    assert verdict(values)["status"] == "inconclusive_coverage"


@pytest.mark.parametrize("mode", ["pair_integrity", "fixture_review"])
def test_stage_audit_errors_survive_empty_provider_error_list(tmp_path, monkeypatch, mode):
    from aftersalesagent.evaluation.e3_comparison import execution

    plan, episodes, primary = fixture()
    monkeypatch.setattr(execution, "load", lambda root, split: ([], {}))
    monkeypatch.setattr(execution, "planned_pairs", lambda root, split: plan)
    monkeypatch.setattr(execution, "evidence", lambda folder, cases: (episodes, primary, True))
    monkeypatch.setattr(
        execution,
        "audit_pairs",
        lambda *args: ["paired_initial_conditions_differ"] if mode == "pair_integrity" else [],
    )
    (tmp_path / "provider_ledger.jsonl").write_text('{"error":null}\n', encoding="utf-8")
    stage = tmp_path / "development/review"
    stage.mkdir(parents=True)
    if mode == "fixture_review":
        (stage / "adjudication.json").write_text(
            '{"reviewer_kind":"test_fixture"}', encoding="utf-8"
        )
    result = execution.decide(ROOT, tmp_path, "development")
    assert result["status"] == "inconclusive_technical"
    assert (
        "paired_initial_conditions_differ"
        if mode == "pair_integrity"
        else "invalid_live_review_provenance"
    ) in result["reasons"]
