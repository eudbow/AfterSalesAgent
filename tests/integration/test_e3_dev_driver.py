"""新 E3 开发驱动器：真实 SQLite/MCP，脚本模型只验证工程和输入隔离。"""

import asyncio
import json
from pathlib import Path

import pytest

from aftersalesagent.evaluation.e3_dev_runner import WorkflowReplay, asks, normalise, run_case
from aftersalesagent.evaluation.e3_dev_suite import build, load
from aftersalesagent.policies.application import load_application_policy
from aftersalesagent.schemas.contracts import RunContext, RunLimits

ROOT = Path(__file__).resolve().parents[2]


def test_new_development_suite_has_separated_inputs_and_prior_user_exclusion():
    cases, manifest = load(ROOT)
    assert build(ROOT) == manifest
    assert len(cases) == 12 and manifest["episodes"] == 48
    assert manifest["repetitions"] == 2 and manifest["split"] == "development_only"
    assert len({c["plan"]["family"] for c in cases}) == 12
    for case in cases:
        assert set(case["public"]) == {"task_id", "first_user_message"}
        assert case["plan"]["source_user_id"] not in manifest["excluded_prior_users"]
        assert case["plan"]["final"]["arguments"] == case["criteria"]["arguments"]


@pytest.mark.parametrize("index", range(12))
def test_new_driver_and_future_message_privacy(index, tmp_path, retail_template):
    case = load(ROOT)[0][index]
    seen = []

    class SpyReplay(WorkflowReplay):
        async def generate(self, request):
            seen.append(request.model_copy(deep=True))
            return await super().generate(request)

    async def execute():
        return await run_case(
            case,
            folder=tmp_path / "episode",
            template=retail_template,
            factory=lambda: SpyReplay(case["plan"]),
            policy=load_application_policy(ROOT),
            context=RunContext(
                environment_id="assigned",
                limits=RunLimits.model_validate_json(
                    (ROOT / "configs/m3_live_limits.json").read_bytes()
                ),
            ),
            group="S1" if index % 2 else "S0",
            repetition=1,
            root=ROOT,
        )

    row = asyncio.run(execute())
    assert row["verification"]["success"], row
    first = seen[0]
    assert [m.content for m in first.messages if m.role == "user"] == [
        case["public"]["first_user_message"]
    ]
    first_content = "\n".join(m.content for m in first.messages)
    for step in case["plan"]["steps"]:
        assert step["reply"] not in first_content
    assert "expected_arguments" not in first_content and "source_user_id" not in first_content
    assert row["process"]["invalid_arguments"] == 0
    assert row["verification"]["checks"]["no_unconfirmed_business_changes"]
    conversation = json.loads((tmp_path / "episode/conversation.json").read_text("utf-8"))
    assert conversation[-1]["event"]["kind"] == "confirmation"
    assert conversation[-1]["event"]["approved"]


@pytest.mark.parametrize(
    ("slot", "text"),
    [
        ("zip", "Please provide the postal code?"),
        ("items", "具体要退哪些商品？"),
        ("reason", "请告诉我取消原因。"),
        ("payment", "Which refund method would you like?"),
    ],
)
def test_clarification_triggers_are_bilingual(slot, text):
    assert asks(text, slot)
    assert not asks("已收到您的请求。", slot)


def test_item_list_order_is_semantically_normalised_only_for_scoring():
    original = {"item_ids": ["b", "a"], "payment_method_id": "card"}
    assert normalise(original)["item_ids"] == ["a", "b"]
    assert original["item_ids"] == ["b", "a"]
