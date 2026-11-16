"""Three bounded real-model development scenarios; append every API attempt to ledger."""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from aftersalesagent.business.database import clone_database
from aftersalesagent.business.service import RetailService
from aftersalesagent.harness.journal import Journal
from aftersalesagent.knowledge.retail import RetailPolicyIndex
from aftersalesagent.local_config import load_local_config, options_from_settings
from aftersalesagent.models.deepseek import DeepSeekBackend
from aftersalesagent.policies.application import load_retail_core_policy
from aftersalesagent.runtime.langgraph_runtime import LangGraphRuntime
from aftersalesagent.schemas.contracts import ConfirmationEvent, MessageEvent, RunContext, RunLimits
from aftersalesagent.tools.registry import ToolRegistry

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/results/retail_agent_rag_v1"
LEDGER = BASE / "api_ledger.json"
PILOT_DIR = BASE / "pilot"
CASES = [
    dict(
        id="D01-return-new-card",
        expected="return requested",
        order="#W1023987",
        prompt=(
            "我是 Sophia Garcia，邮编 78263。订单 #W1023987 已收到，"
            "我想退订单里的第一件商品。能退到一张新的银行卡吗？"
            "如果不行，请用原支付方式提交退货申请。"
        ),
    ),
    dict(
        id="D02-processed-cancel",
        expected="processed",
        order="#W1770559",
        prompt=(
            "我是 Isabella Thomas，邮编 28243。请取消订单 #W1770559，"
            "原因是我不再需要了；如果不能取消，请解释依据。"
        ),
    ),
    dict(
        id="D03-return-timing",
        expected="no_write",
        order="#W1023987",
        prompt=(
            "我是 Sophia Garcia，邮编 78263。我想知道订单 #W1023987 申请退货后，"
            "退款一定几天到账？请给出政策依据。"
        ),
    ),
]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", "utf-8")
    tmp.replace(path)


class AccountedBackend:
    def __init__(self, backend, case_id):
        self.backend, self.case_id = backend, case_id

    async def generate(self, request):
        data = json.loads(LEDGER.read_text("utf-8"))
        if data["completed_api_attempts"] + data["reserved_api_attempts"] >= 720:
            raise RuntimeError("API attempt ceiling reached")
        item = data["runs"][-1]
        if item["id"] != self.case_id or item["actual_attempts"] >= item["max_attempts"]:
            raise RuntimeError("Case API attempt ceiling reached")
        data["reserved_api_attempts"] += 1
        item["actual_attempts"] += 1
        write_json(LEDGER, data)
        try:
            response = await self.backend.generate(request)
            data = json.loads(LEDGER.read_text("utf-8"))
            item = data["runs"][-1]
            item["input_tokens"] += response.input_tokens or 0
            item["output_tokens"] += response.output_tokens or 0
            write_json(LEDGER, data)
            return response
        finally:
            data = json.loads(LEDGER.read_text("utf-8"))
            data["reserved_api_attempts"] -= 1
            data["completed_api_attempts"] += 1
            write_json(LEDGER, data)


async def run_case(case, options, secret, index):
    path = PILOT_DIR / (case["id"] + ".sqlite")
    if path.exists():
        raise RuntimeError("Pilot case already has a database; no implicit rerun")
    store = clone_database(ROOT / ".local/databases/retail_template.sqlite", path)
    service = RetailService(store)
    backend = DeepSeekBackend(options, secret, allow_network=True)
    journal = Journal(PILOT_DIR / (case["id"] + "-journal.sqlite"))
    policy = load_retail_core_policy()
    runtime = LangGraphRuntime(
        AccountedBackend(backend, case["id"]),
        service,
        policy_text=policy,
        registry=ToolRegistry(service, retail_index=index),
        journal=journal,
        trace_directory=PILOT_DIR / "traces",
    )
    try:
        view = await runtime.start(
            RunContext(
                environment_id=service.environment_id,
                backend="deepseek",
                model_name=options.model,
                knowledge_version="retail_sim_v1",
                knowledge_fingerprint=index.fingerprint,
                max_output_tokens=1024,
                limits=RunLimits(
                    max_model_calls=12,
                    max_tool_calls=20,
                    max_active_seconds=120,
                max_tokens=48000,
                    max_turn_model_calls=8,
                    max_turn_tool_calls=12,
                    max_turn_active_seconds=90,
                max_turn_tokens=24000,
                ),
            )
        )
        result = await runtime.advance(
            view.session_id, MessageEvent(event_id=case["id"] + "-message", content=case["prompt"])
        )
        first_status = result.status
        if result.status == "waiting_confirmation" and case["expected"] == "return requested":
            action = result.session.pending_action
            result = await runtime.advance(
                view.session_id,
                ConfirmationEvent(
                    event_id=case["id"] + "-confirm", action_id=action.action_id, approved=True
                ),
            )
        frame, _ = await runtime._snapshot(view.session_id)
        with store.transaction() as connection:
            order = store.order(connection, case["order"])
        return dict(
            id=case["id"],
            first_status=first_status,
            final_status=result.status,
            reply=result.reply,
            sources=result.sources,
            order_status=order["status"],
            model_calls=journal.usage(view.session_id)["model_calls"],
            tool_calls=journal.usage(view.session_id)["tool_calls"],
            trace=str(PILOT_DIR / "traces"),
            raw_response=frame.response.model_dump(mode="json") if frame.response else None,
        )
    finally:
        await backend.aclose()
        journal.close()
        store.close()


async def main():
    global PILOT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--label", default="v1")
    parser.add_argument("--enable-paid-api", action="store_true")
    args = parser.parse_args()
    if not args.enable_paid_api:
        parser.error("Explicit paid API flag required")
    if not args.label.isalnum():
        parser.error("Label must be alphanumeric")
    PILOT_DIR = BASE / ("pilot_" + args.label)
    cases = [dict(case, id=case["id"] + "-" + args.label) for case in CASES]
    settings, secret = load_local_config(args.env_file)
    options = options_from_settings(settings)
    if not secret.get_secret_value():
        parser.error("Local DeepSeek key missing")
    index = RetailPolicyIndex(method="bm25")
    try:
        frozen = dict(
            cases=cases,
            model=options.model,
            index_fingerprint=index.fingerprint,
            policy_mode="tool_rag_bm25",
            temperature=options.temperature,
        )
        frozen["sha256"] = hashlib.sha256(
            json.dumps(frozen, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        target = PILOT_DIR / "frozen.json"
        if target.exists() and json.loads(target.read_text("utf-8")) != frozen:
            raise RuntimeError("Pilot protocol changed")
        write_json(target, frozen)
        ledger = json.loads(LEDGER.read_text("utf-8"))
        if ledger["actual_trajectories"] + len(cases) > 60:
            raise RuntimeError("Trajectory ceiling reached")
        for case in cases:
            result_path = PILOT_DIR / (case["id"] + ".json")
            if result_path.exists():
                continue
            ledger = json.loads(LEDGER.read_text("utf-8"))
            ledger["actual_trajectories"] += 1
            ledger["runs"].append(
                dict(
                    kind="Retail business pilot",
                    id=case["id"],
                    status="running",
                    max_attempts=12,
                    actual_attempts=0,
                    input_tokens=0,
                    output_tokens=0,
                )
            )
            write_json(LEDGER, ledger)
            try:
                result = await run_case(case, options, secret, index)
                write_json(result_path, result)
                status = "completed"
            except Exception as exc:
                status = "failed"
                result = dict(id=case["id"], error=type(exc).__name__)
                write_json(result_path, result)
            ledger = json.loads(LEDGER.read_text("utf-8"))
            ledger["runs"][-1]["status"] = status
            write_json(LEDGER, ledger)
            print(
                json.dumps(
                    {
                        k: result.get(k)
                        for k in (
                            "id",
                            "first_status",
                            "final_status",
                            "order_status",
                            "model_calls",
                            "tool_calls",
                            "error",
                        )
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if status == "failed":
                break
    finally:
        index.close()


if __name__ == "__main__":
    asyncio.run(main())
