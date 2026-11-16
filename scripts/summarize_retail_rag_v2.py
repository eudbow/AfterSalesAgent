"""Recompute a complete v2 batch score from saved DB assessments and review labels."""

import argparse
import json
import statistics
from pathlib import Path

from summarize_retail_rag_comparison import parameters_correct

from aftersalesagent.business.database import RetailStore
from aftersalesagent.harness.journal import Journal

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/results/retail_agent_rag_v2"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True)
    args = parser.parse_args()
    batch = (BASE / args.batch).resolve()
    if batch.parent != BASE.resolve():
        parser.error("Batch escapes results directory")
    labels = json.loads((batch / "review_labels.json").read_text("utf-8"))
    protocol = json.loads((batch / "protocol.json").read_text("utf-8"))
    ids = {f"R{i:02}" for i in range(1, 25)}
    if set(protocol["cases"]) != ids or set(labels["reviewed_ids"]) != ids:
        raise ValueError("Complete 24-case batch and explicit review coverage required")
    runs, rows, usage = (
        [],
        [],
        {"model_calls": 0, "retry_events": 0, "repair_events": 0, "cache_hit": 0, "cache_miss": 0},
    )
    for cid in sorted(ids):
        run = json.loads((batch / f"{cid}-tool_rag.json").read_text("utf-8"))
        store = RetailStore(batch / f"{cid}-tool_rag.sqlite")
        try:
            with store.transaction() as connection:
                actual_order = store.order(connection, run["case"]["order_id"])
            if actual_order != run["after_order"]:
                raise ValueError("Saved database/result mismatch: " + cid)
        finally:
            store.close()
        business = run["assessment"]["write_boundary_correct"] and parameters_correct(run)
        issues = labels["critical_issues"].get(cid, [])
        rows.append(
            {
                "id": cid,
                "business_pass": business,
                "strict_pass": business and not issues,
                "issues": issues,
            }
        )
        runs.append(run)
        journal = Journal(batch / f"{cid}-tool_rag-journal.sqlite")
        try:
            events = journal.events(run["session_id"])
        finally:
            journal.close()
        usage["retry_events"] += sum(e.kind == "retry" for e in events)
        usage["repair_events"] += sum(e.payload.get("code") == "policy_repair" for e in events)
        for event in events:
            if event.kind == "model_response":
                usage["model_calls"] += 1
                meta = event.payload.get("provider_metadata", {})
                usage["cache_hit"] += meta.get("prompt_cache_hit_tokens") or 0
                usage["cache_miss"] += meta.get("prompt_cache_miss_tokens") or 0
    ledger = json.loads((batch / "api_ledger.json").read_text("utf-8"))
    summary = {
        "batch": args.batch,
        "n": 24,
        "business_pass": sum(r["business_pass"] for r in rows),
        "strict_pass": sum(r["strict_pass"] for r in rows),
        "reviewer": labels["reviewer"],
        "api_attempts": ledger["completed_api_attempts"],
        "input_tokens": sum(r["input_tokens"] for r in ledger["runs"]),
        "output_tokens": sum(r["output_tokens"] for r in ledger["runs"]),
        "search_calls": sum(r["assessment"]["search_count"] for r in runs),
        "median_elapsed_seconds": statistics.median(r["elapsed_seconds"] for r in runs),
        "usage": usage,
        "rows": rows,
    }
    (batch / "review_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", "utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
