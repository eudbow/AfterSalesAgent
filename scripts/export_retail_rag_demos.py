"""Replay three recorded DeepSeek pilot traces offline against fresh Retail orders."""

import argparse
import asyncio
import json
from pathlib import Path

from aftersalesagent.business.database import clone_database
from aftersalesagent.business.service import RetailService
from aftersalesagent.harness.journal import Journal
from aftersalesagent.knowledge.retail import RetailPolicyIndex
from aftersalesagent.models.mock import MockBackend
from aftersalesagent.policies.application import load_retail_core_policy
from aftersalesagent.runtime.langgraph_runtime import LangGraphRuntime
from aftersalesagent.schemas.contracts import (
    ConfirmationEvent,
    MessageEvent,
    ModelResponse,
    RunContext,
    RunLimits,
    TurnResult,
)
from aftersalesagent.tools.registry import ToolRegistry

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/results/retail_agent_rag_v1"
SOURCE = BASE / "pilot_v3"
OUTPUT = BASE / "demo_replay_v2"
CASES = {
    "D01-return-new-card-v3": ("#W1023987", "return requested"),
    "D02-processed-cancel-v3": ("#W1770559", "processed"),
    "D03-return-timing-v3": ("#W1023987", "delivered"),
}


def order_snapshot(store, order_id):
    with store.transaction() as connection:
        return store.order(connection, order_id)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", "utf-8")


def recorded_inputs(case_id):
    path = SOURCE / (case_id + "-journal.sqlite")
    journal = Journal(path)
    try:
        sessions = journal.connection.execute("SELECT session_id FROM usage").fetchall()
        if len(sessions) != 1:
            raise ValueError("Pilot journal must contain one session")
        sid = sessions[0][0]
        events = journal.events(sid)
        messages = [event for event in events if event.kind == "user_message"]
        responses = [event for event in events if event.kind == "model_response"]
        if len(messages) != 1 or not responses:
            raise ValueError("Pilot trace lacks initial message or model responses")
        original = journal.request(sid, messages[0].payload["event_id"])
        return {
            "source_session_id": sid,
            "prompt": messages[0].payload["content"],
            "responses": [
                ModelResponse.model_validate(
                    {key: value for key, value in e.payload.items() if key != "request_event_id"}
                )
                for e in responses
            ],
            "original_first_result": TurnResult.model_validate_json(original["result_json"]),
            "source_trace": str(path),
        }
    finally:
        journal.close()


async def replay(case_id, order_id, expected_status, index):
    input_data = recorded_inputs(case_id)
    path = OUTPUT / (case_id + ".sqlite")
    result_path = OUTPUT / (case_id + ".json")
    if path.exists() or result_path.exists():
        raise ValueError("Replay artifact exists; no implicit overwrite")
    store = clone_database(ROOT / ".local/databases/retail_template.sqlite", path)
    journal = Journal(OUTPUT / (case_id + "-journal.sqlite"))
    service = RetailService(store)
    backend = MockBackend(input_data["responses"])
    runtime = LangGraphRuntime(
        backend,
        service,
        policy_text=load_retail_core_policy(),
        registry=ToolRegistry(service, retail_index=index),
        journal=journal,
        trace_directory=OUTPUT / "traces",
    )
    try:
        before = order_snapshot(store, order_id)
        view = await runtime.start(
            RunContext(
                environment_id=service.environment_id,
                backend="mock",
                knowledge_version="retail_sim_v1",
                knowledge_fingerprint=index.fingerprint,
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
        first = await runtime.advance(
            view.session_id,
            MessageEvent(event_id=case_id + "-message", content=input_data["prompt"]),
        )
        before_confirmation = order_snapshot(store, order_id)
        action = first.session.pending_action
        final = first
        if action is not None and expected_status == "return requested":
            final = await runtime.advance(
                view.session_id,
                ConfirmationEvent(
                    event_id=case_id + "-confirm", action_id=action.action_id, approved=True
                ),
            )
        after = order_snapshot(store, order_id)
        events = journal.events(view.session_id)
        searches = [
            e.payload
            for e in events
            if e.kind == "tool_call" and e.payload["name"] == "search_retail_policy"
        ]
        evidence = [
            item
            for event in events
            if event.kind == "tool_result" and isinstance(event.payload.get("data"), dict)
            for item in event.payload["data"].get("evidence", [])
        ]
        result = {
            "mode": "offline replay of recorded real DeepSeek model responses; zero new API calls",
            "source_trace": input_data["source_trace"],
            "source_session_id": input_data["source_session_id"],
            "recorded_model_responses": len(input_data["responses"]),
            "case_id": case_id,
            "user_message": input_data["prompt"],
            "order_id": order_id,
            "order_before": before,
            "policy_searches": searches,
            "policy_evidence": evidence,
            "original_public_reply": input_data["original_first_result"].reply,
            "replay_first_reply": first.reply,
            "replay_first_sources": first.sources,
            "citation_check_reason": first.termination_reason,
            "pending_action": action.model_dump(mode="json") if action else None,
            "order_before_confirmation": before_confirmation,
            "confirmation_event": (
                {"action_id": action.action_id, "approved": True}
                if action and expected_status == "return requested"
                else None
            ),
            "final_reply": final.reply,
            "final_status": final.status,
            "order_after": after,
            "state_change_count": sum(e.kind == "state_change" for e in events),
        }
        if after["status"] != expected_status:
            raise AssertionError("Replay order status mismatch")
        if action and before_confirmation != before:
            raise AssertionError("Order changed before confirmation")
        write_json(result_path, result)
        return result
    finally:
        journal.close()
        store.close()


async def main():
    global OUTPUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    OUTPUT = args.output_dir.resolve()
    if not OUTPUT.is_relative_to(BASE.resolve()):
        parser.error("Output directory must stay under the Retail results directory")
    index = RetailPolicyIndex()
    try:
        for case_id, (order_id, expected) in CASES.items():
            result = await replay(case_id, order_id, expected, index)
            print(
                json.dumps(
                    {
                        "case": case_id,
                        "status": result["order_after"]["status"],
                        "searches": len(result["policy_searches"]),
                        "sources": [s["chunk_id"] for s in result["replay_first_sources"]],
                        "state_changes": result["state_change_count"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    finally:
        index.close()


if __name__ == "__main__":
    asyncio.run(main())
