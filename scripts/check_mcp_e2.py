"""E2离线验收：真实stdio与Native语义/证据对照及有限耗时测量；不加载.env。"""

import asyncio
import hashlib
import json
import math
import statistics
import time
import uuid
from pathlib import Path

from aftersalesagent.business.database import clone_database, import_pinned_database
from aftersalesagent.business.service import RetailService
from aftersalesagent.business.source_audit import verify_source
from aftersalesagent.evaluation.e1_suite import load
from aftersalesagent.harness.journal import Journal
from aftersalesagent.knowledge.corpus import load_corpus
from aftersalesagent.knowledge.retrieval import SQLiteKnowledgeIndex
from aftersalesagent.mcp_transport.adapters import MCPKnowledgeProvider, MCPRetailProvider
from aftersalesagent.models.scenarios import SCENARIOS, ScenarioMockBackend
from aftersalesagent.policies.application import load_application_policy
from aftersalesagent.runtime.langgraph_runtime import LangGraphRuntime
from aftersalesagent.schemas.contracts import (
    ConfirmationEvent,
    MessageEvent,
    RunContext,
    ToolCall,
    ToolContext,
)
from aftersalesagent.tools.registry import ToolRegistry

ROOT = Path(__file__).resolve().parents[1]


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", "utf-8")


def normalize(value, volatile):
    if isinstance(value, dict):
        return {
            k: "<expiry>" if k == "expires_at" else normalize(v, volatile) for k, v in value.items()
        }
    if isinstance(value, list):
        return [normalize(v, volatile) for v in value]
    if isinstance(value, str) and value in volatile:
        return volatile[value]
    return value


def tool_context(runtime, sid, key):
    view = runtime.service.view(sid)
    return ToolContext(
        session_id=sid,
        environment_id=view.environment_id,
        authenticated_user_id=view.authenticated_user_id,
        idempotency_key=key,
    )


def timing(samples):
    values = sorted(samples)
    return {
        "n": len(values),
        "median_ms": statistics.median(values) * 1000,
        "p95_ms": values[math.ceil(len(values) * 0.95) - 1] * 1000,
    }


async def business(folder, template):
    pairs, warm = [], {"native": [], "mcp": []}
    for scenario in ("query", "cancel", "address", "return"):
        active, rows, volatile = {}, {}, {}
        try:
            for mode in ("native", "mcp"):
                directory = folder / scenario / mode
                directory.mkdir(parents=True)
                store = clone_database(template, directory / "environment.sqlite")
                service = RetailService(store)
                journal = Journal(directory / "journal.sqlite")
                provider = (
                    MCPRetailProvider(service, trace_path=directory / "mcp.jsonl")
                    if mode == "mcp"
                    else None
                )
                registry = ToolRegistry(service, provider=provider)
                runtime = LangGraphRuntime(
                    ScenarioMockBackend(scenario),
                    service,
                    registry=registry,
                    policy_text=load_application_policy(ROOT),
                    journal=journal,
                    trace_directory=directory,
                )
                session = await runtime.start(
                    RunContext(environment_id=service.environment_id, tool_transport=mode)
                )
                sid = session.session_id
                active[mode] = (runtime, sid)
                volatile[sid] = "<session>"
                cold = None
                if provider:
                    started = time.perf_counter()
                    await provider.connection(sid).connect()
                    cold = time.perf_counter() - started
                    discovery = provider.connection(sid).discovery
                    for spec in service.specs():
                        remote = next(t for t in discovery if t.name == spec.name)
                        assert remote.input_schema == spec.arguments_schema
                        assert remote.description == spec.description
                        assert (remote.meta or {})["aftersales/effect"] == spec.effect
                started = time.perf_counter()
                event = MessageEvent(event_id="message-1", content=SCENARIOS[scenario]["prompt"])
                result = await runtime.advance(sid, event)
                assert await runtime.advance(sid, event) == result
                oid = "#W5490111" if scenario == "return" else "#W5056519"
                with store.transaction() as connection:
                    before = store.order(connection, oid)
                if scenario != "query":
                    assert result.status == "waiting_confirmation"
                    action = result.session.pending_action
                    assert action is not None
                    volatile[action.action_id] = "<action>"
                    confirmation = ConfirmationEvent(
                        event_id="confirm-1", action_id=action.action_id, approved=True
                    )
                    confirmed = await runtime.advance(sid, confirmation)
                    assert confirmed.status == "completed"
                    assert await runtime.advance(sid, confirmation) == confirmed
                else:
                    assert result.status == "completed"
                with store.transaction() as connection:
                    after = store.order(connection, oid)
                observations = [e.payload for e in journal.events(sid) if e.kind == "tool_result"]
                rows[mode] = {
                    "proposal_status": result.status,
                    "before_confirmation": before,
                    "final_order": after,
                    "tool_results": observations,
                    "model_calls": journal.usage(sid)["model_calls"],
                    "cold_discovery_seconds": cold,
                    "warm_flow_seconds": time.perf_counter() - started,
                }
                journal.export(sid, directory)
            for field in (
                "proposal_status",
                "before_confirmation",
                "final_order",
                "tool_results",
                "model_calls",
            ):
                assert normalize(rows["native"][field], volatile) == (
                    normalize(rows["mcp"][field], volatile)
                ), (scenario, field)
            errors = []
            if scenario == "query":
                # 同一问题在两个独立环境上交替测量，单独记录已建连开销。
                for i in range(30):
                    call = ToolCall(
                        call_id=f"timing-{i}",
                        name="get_order_details",
                        arguments={"order_id": "#W5056519"},
                    )
                    observations = []
                    modes = ("native", "mcp") if i % 2 == 0 else ("mcp", "native")
                    for mode in modes:
                        runtime, sid = active[mode]
                        started = time.perf_counter()
                        actual = await runtime.registry.invoke(
                            call, tool_context(runtime, sid, f"timing-{i}")
                        )
                        warm[mode].append(time.perf_counter() - started)
                        observations.append(actual)
                    assert observations[0] == observations[1]
                for i, arguments in enumerate(
                    (
                        {},
                        {"order_id": "#W5490111"},
                        {"order_id": "#NO_SUCH_ORDER"},
                        {"order_id": "#W5056519", "approved": True},
                    )
                ):
                    call = ToolCall(
                        call_id=f"error-{i}", name="get_order_details", arguments=arguments
                    )
                    results = []
                    for mode in ("native", "mcp"):
                        runtime, sid = active[mode]
                        result = await runtime.registry.invoke(
                            call, tool_context(runtime, sid, f"error-{i}")
                        )
                        results.append(result)
                    assert results[0] == results[1] and results[0].error is not None
                    errors.append(results[0].model_dump(mode="json"))
            pairs.append({"scenario": scenario, "equal": True, "routes": rows, "errors": errors})
        finally:
            for runtime, _ in active.values():
                await runtime.registry.aclose()
                runtime.journal.close()
                runtime.service.store.close()
    save(folder / "business.json", pairs)
    return {
        "workflow_pairs": len(pairs),
        "workflow_semantics_equal": True,
        "schema_tools_equal": 10,
        "error_pairs_equal": 4,
        "warm_order_lookup": {mode: timing(v) for mode, v in warm.items()},
        "cold_discovery_seconds": [p["routes"]["mcp"]["cold_discovery_seconds"] for p in pairs],
        "normalization": "session/action random IDs and proposal expiry only; order IDs unchanged",
    }


async def knowledge(folder):
    chunks, _ = load_corpus(ROOT)
    native = SQLiteKnowledgeIndex(chunks)
    remote = MCPKnowledgeProvider(
        SQLiteKnowledgeIndex(chunks), trace_path=folder / "knowledge_mcp.jsonl"
    )
    rows = []
    try:
        for split in ("dev", "validation", "test"):
            cases, _ = load(ROOT, split)
            for case in cases:
                p = case["public"]
                for full in (False, True):
                    expected = await native.provide(p["question"], p["filters"], 5, full=full)
                    actual = await remote.provide(p["question"], p["filters"], 5, full=full)
                    assert expected == actual
                    rows.append(
                        {
                            "task": p["task_id"],
                            "split": split,
                            "mode": "full" if full else "rag",
                            "ids": [c.chunk_id for c in actual],
                            "equal": True,
                        }
                    )
        # 验证线上的分数就是本地BM25分数，而非适配器制造的占位数。
        query = "银行卡退货凭证"
        result = await remote.remote.call(
            "search_policy", {"query": query, "filters": {"scope": "ikea_cn"}, "top_k": 5}
        )
        assert [
            (c.chunk_id, score) for c, score in native.ranked(query, {"scope": "ikea_cn"}, 5)
        ] == [(e["chunk"]["chunk_id"], e["score"]) for e in result["evidence"]]
        protocol = remote.remote.protocol_version
    finally:
        await remote.aclose()
        remote.close()
        native.close()
    save(folder / "knowledge.json", rows)
    return {
        "evidence_pairs": len(rows),
        "exact_chunks_equal": True,
        "bm25_scores_equal": True,
        "protocol_version": protocol,
        "note": "Existing E1 tasks reused as compatibility regression, not new held-out results",
    }


async def main():
    run_id = "e2_" + uuid.uuid4().hex[:10]
    folder = ROOT / ".local/e2_runs" / run_id
    folder.mkdir(parents=True)
    source = [
        *ROOT.joinpath("src").rglob("*.py"),
        Path(__file__).resolve(),
        ROOT / "uv.lock",
        ROOT / "pyproject.toml",
    ]
    snapshot = {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in source
    }
    save(folder / "config.json", snapshot)
    template = ROOT / ".local/databases/retail_template.sqlite"
    if not template.exists():
        import_pinned_database(ROOT, template).close()
    business_result = await business(folder, template)
    knowledge_result = await knowledge(folder)
    before_path = ROOT / ".local/results/e2_before_hashes.json"
    protected = {}
    if before_path.exists():
        before = json.loads(before_path.read_text("utf-8"))
        for name, sha in before.items():
            if name.startswith(("data/", "../")) or name in {
                ".local/results/e1_acceptance.json",
                ".local/results/e1_final_freeze.json",
                "src/aftersalesagent/business/service.py",
            }:
                protected[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha
    assert all(protected.values())
    assert not verify_source(ROOT)
    assert all(
        hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == sha for p, sha in snapshot.items()
    )
    report = {
        "run_id": run_id,
        "passed": True,
        "mode": "offline_native_mcp_conformance",
        "business": business_result,
        "knowledge": knowledge_result,
        "protected_files_unchanged": protected,
        "source_freeze_unchanged": True,
        "real_model_calls": 0,
        "gpu_experiments": 0,
        "notes": [
            "Small local sequential timing, not a production capacity benchmark",
            "Confirmation commits remain in the original trusted application entry",
            "Cold startup and warm calls reported separately; no claim of improved model quality",
        ],
    }
    save(folder / "summary.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("Summary:", folder / "summary.json")


if __name__ == "__main__":
    asyncio.run(main())
