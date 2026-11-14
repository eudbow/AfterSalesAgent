"""真正Runtime/MCP/SQLite配对和盲表绑定；只用未来脚本夹具，不计模型分数。"""

import asyncio
import json
from pathlib import Path

import pytest

from aftersalesagent.evaluation.e3_candidates.protocol import PreparationUser, load_protocol
from aftersalesagent.evaluation.e3_comparison.execution import registry, unlocked
from aftersalesagent.evaluation.e3_comparison.materials import load
from aftersalesagent.evaluation.e3_comparison.review import (
    evidence,
    export_review,
    import_review,
    primary_sequence,
)
from aftersalesagent.evaluation.e3_v2.replay import ProtocolReplay
from aftersalesagent.evaluation.e3_v2.runner import run_case
from aftersalesagent.policies.application import load_application_policy
from aftersalesagent.schemas.contracts import RunContext

ROOT = Path(__file__).resolve().parents[2]


def test_holdouts_are_locked_without_previous_stage_decision(tmp_path):
    for split in ("validation", "test"):
        with pytest.raises(ValueError):
            unlocked(ROOT, tmp_path, split)


def test_real_pair_uses_correct_skill_and_review_binding(tmp_path, retail_template):
    case = load(ROOT, "development")[0][14]  # en→zh、缺邮编追问
    reg = registry(ROOT, tmp_path / "experimental.sqlite")

    async def run():
        for group, version in (("S1", "1.0.0"), ("S2", "1.0.1")):
            await run_case(
                case,
                folder=tmp_path / f"{case['public'].task_id}_r1_{group}",
                template=retail_template,
                backend_factory=lambda: ProtocolReplay(case["script"], variant="direct"),
                user_factory=PreparationUser,
                skill_provider=reg,
                group=group,
                transport="mcp",
                policy_text=load_application_policy(ROOT),
                protocol=load_protocol(ROOT),
                context=RunContext(
                    environment_id="assigned",
                    skill_id="after-sales-workflow",
                    skill_version=version,
                ),
            )

    asyncio.run(run())
    for group, version in (("S1", "1.0.0"), ("S2", "1.0.1")):
        folder = tmp_path / f"{case['public'].task_id}_r1_{group}"
        events = [json.loads(s) for s in (folder / "trace.jsonl").read_text("utf-8").splitlines()]
        first = next(e["payload"] for e in events if e["kind"] == "model_request")
        assert first["skill_version"] == version
        assert primary_sequence(folder, case["anchor"]) is not None
    exported = export_review(tmp_path, [case])
    assert exported["items"] == 2
    sheet = json.loads((tmp_path / "review/sheet.json").read_text("utf-8"))
    assert all("group" not in item and "episode_id" not in item for item in sheet["items"])
    for item in sheet["items"]:
        item["annotation"] = {
            "label": item["expected_language"],
            "assessment": "pass",
            "rationale": "offline fixture only",
            "false_execution_claim": False,
            "claim_rationale": "fixture asks missing zip; no execution claim",
        }
    target = tmp_path / "review/fixture_sheet.json"
    target.write_text(json.dumps(sheet), encoding="utf-8")
    import_review(
        tmp_path, reviewed_sheet=target, reviewer="offline fixture", reviewer_kind="test_fixture"
    )
    episodes, primary, complete = evidence(tmp_path, [case])
    assert len(episodes) == 2 and complete
    assert primary[0]["S1"] == primary[0]["S2"] == "pass"
    assert primary[1]["S1"] is None  # 未运行r2不能偷偷删除分母。
    with pytest.raises(ValueError):
        import_review(
            tmp_path, reviewed_sheet=target, reviewer="fixture", reviewer_kind="test_fixture"
        )


@pytest.mark.parametrize("mode", ["text", "source", "foreign_id"])
def test_review_rejects_changed_text_source_or_foreign_id(tmp_path, mode):
    from aftersalesagent.evaluation.e3_comparison.review import export_review, import_review

    folder = tmp_path / "case_r1_S1"
    folder.mkdir()
    row = {
        "task_id": "case",
        "scoring": {
            "language": {
                "records": [
                    {
                        "sequence": 1,
                        "expected_language": "en",
                        "user_context": "Which zip?",
                        "text": "Please provide the new zip code.",
                        "allowed_literals": [],
                    }
                ]
            }
        },
    }
    (folder / "result.json").write_text(json.dumps(row), encoding="utf-8")
    for name in ("conversation", "initial_state", "final_state"):
        (folder / (name + ".json")).write_text("{}", encoding="utf-8")
    (folder / "trace.jsonl").write_text(
        json.dumps(
            {
                "sequence": 1,
                "kind": "model_response",
                "payload": {"content": "Please provide the new zip code."},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    export_review(tmp_path, [])
    sheet = json.loads((tmp_path / "review/sheet.json").read_text("utf-8"))
    sheet["items"][0]["annotation"] = {
        "label": "en",
        "assessment": "pass",
        "rationale": "fixture",
        "false_execution_claim": False,
        "claim_rationale": "fixture asks zip",
    }
    if mode == "text":
        sheet["items"][0]["text"] = "different text"
    elif mode == "foreign_id":
        sheet["items"][0]["review_id"] = "foreign"
    else:
        with (folder / "result.json").open("a", encoding="utf-8") as stream:
            stream.write(" ")
    path = tmp_path / "review/annotated.json"
    path.write_text(json.dumps(sheet), encoding="utf-8")
    with pytest.raises(ValueError):
        import_review(
            tmp_path, reviewed_sheet=path, reviewer="fixture", reviewer_kind="test_fixture"
        )
