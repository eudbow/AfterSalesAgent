"""Recompute the frozen comparison report from immutable per-run artifacts."""

import hashlib
import json
import statistics
from pathlib import Path

from aftersalesagent.harness.journal import Journal
from aftersalesagent.knowledge.retail import RetailPolicyIndex

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/results/retail_agent_rag_v1"
OUT = BASE / "comparison_v1"


def read(path):
    return json.loads(path.read_text("utf-8"))


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", "utf-8")


def parameters_correct(run):
    case, action, before = run["id"], run["action"], run["before_order"]
    if case not in {"R01", "R02", "R13"}:
        return action is None
    if not action or action["order_id"] != run["case"]["order_id"]:
        return False
    args = action["arguments"]
    if case == "R01":
        return args.get("reason") == "no longer needed"
    if case == "R02":
        return all(
            args.get(key) == value
            for key, value in {
                "address1": "10 Pine Street",
                "address2": "Apt 2",
                "city": "Seattle",
                "state": "WA",
                "zip": "98101",
                "country": "USA",
            }.items()
        )
    first = before["items"][0]["item_id"]
    original = before["payment_history"][0]["payment_method_id"]
    return args.get("item_ids") == [first] and args.get("payment_method_id") == original


def aggregate(runs):
    return {
        "n": len(runs),
        "business_boundary_pass": sum(
            r["assessment"]["write_boundary_correct"] and r["parameters_correct"] for r in runs
        ),
        "review_screen_pass": sum(r["review_screen_pass"] for r in runs),
        "total_reported_tokens": sum(r["usage"]["reported_tokens"] for r in runs),
        "input_tokens": sum(r["input_tokens"] for r in runs),
        "output_tokens": sum(r["output_tokens"] for r in runs),
        "model_calls": sum(r["usage"]["model_calls"] for r in runs),
        "prompt_cache_hit_tokens": sum(r["prompt_cache_hit_tokens"] for r in runs),
        "prompt_cache_miss_tokens": sum(r["prompt_cache_miss_tokens"] for r in runs),
        "retry_events": sum(r["retry_events"] for r in runs),
        "model_error_events": sum(r["model_error_events"] for r in runs),
        "retrieval_calls": sum(r["assessment"]["search_count"] for r in runs),
        "runs_with_answer_citations": sum(bool(r["first_sources"]) for r in runs),
        "median_active_seconds": round(statistics.median(r["elapsed_seconds"] for r in runs), 2),
    }


def main():
    frozen = read(BASE / "scenarios_v1.json")
    digest = hashlib.sha256(
        json.dumps(
            {key: value for key, value in frozen.items() if key != "sha256"},
            sort_keys=True,
            ensure_ascii=False,
        ).encode()
    ).hexdigest()
    if digest != frozen["sha256"]:
        raise ValueError("Frozen scenario contents changed")
    cases = frozen["cases"]
    review = read(OUT / "review_labels_v1.json")
    ledger = read(BASE / "api_ledger.json")
    index = RetailPolicyIndex()
    try:
        runs = {}
        for case in cases:
            for arm in ("full", "tool_rag"):
                key = case["id"] + "-" + arm
                run = read(OUT / (key + ".json"))
                if run["id"] != case["id"] or run["arm"] != arm or run["case"] != case:
                    raise ValueError("Frozen case/result mismatch: " + key)
                for source in run["first_sources"]:
                    chunk = index.by_id[source["chunk_id"]]
                    if (
                        source["text"] != chunk.text
                        or source["scope"] != "retail_sim"
                        or source["version"] != "retail_sim_v1"
                    ):
                        raise ValueError("Citation source mismatch: " + key)
                run["parameters_correct"] = parameters_correct(run)
                journal = Journal(OUT / (key + "-journal.sqlite"))
                try:
                    events = journal.events(run["session_id"])
                finally:
                    journal.close()
                model_responses = [event for event in events if event.kind == "model_response"]
                run["prompt_cache_hit_tokens"] = sum(
                    event.payload.get("provider_metadata", {}).get("prompt_cache_hit_tokens") or 0
                    for event in model_responses
                )
                run["prompt_cache_miss_tokens"] = sum(
                    event.payload.get("provider_metadata", {}).get("prompt_cache_miss_tokens") or 0
                    for event in model_responses
                )
                run["retry_events"] = sum(event.kind == "retry" for event in events)
                run["model_error_events"] = sum(event.kind == "model_error" for event in events)
                run["critical_issues"] = review["critical_issues"].get(key, [])
                run["noncritical_notes"] = review["noncritical_notes"].get(key, [])
                run["review_screen_pass"] = (
                    run["assessment"]["write_boundary_correct"]
                    and run["parameters_correct"]
                    and not run["critical_issues"]
                )
                runs[key] = run
        compared = [item for item in ledger["runs"] if item["kind"] == "Retail frozen comparison"]
        usage = {item["id"]: item for item in compared}
        if len(runs) != 48 or set(runs) != set(usage):
            raise ValueError("Comparison/ledger coverage mismatch")
        for key, run in runs.items():
            item = usage[key]
            if item["status"] != "completed" or item["actual_attempts"] != run["usage"][
                "model_calls"
            ]:
                raise ValueError("Model call/ledger mismatch: " + key)
            run["input_tokens"] = item["input_tokens"]
            run["output_tokens"] = item["output_tokens"]
            if item["input_tokens"] + item["output_tokens"] != run["usage"][
                "reported_tokens"
            ]:
                raise ValueError("Token usage/ledger mismatch: " + key)
        summary = {
            "scenario_sha256": read(BASE / "scenarios_v1.json")["sha256"],
            "protocol": read(OUT / "protocol.json"),
            "reviewer": review["reviewer"],
            "total_trajectories_including_pilots": ledger["actual_trajectories"],
            "total_api_attempts_including_smoke": ledger["completed_api_attempts"],
            "comparison_api_attempts": sum(x["actual_attempts"] for x in compared),
            "comparison": {
                arm: aggregate([r for r in runs.values() if r["arm"] == arm])
                for arm in ("full", "tool_rag")
            },
            "by_split": {
                split: {
                    arm: aggregate(
                        [r for r in runs.values() if r["split"] == split and r["arm"] == arm]
                    )
                    for arm in ("full", "tool_rag")
                }
                for split in ("dev", "test")
            },
            "cases": [
                {
                    "id": case["id"],
                    "split": case["split"],
                    "arms": {
                        arm: {
                            "status": runs[case["id"] + "-" + arm]["first_status"],
                            "business_boundary_pass": (
                                runs[case["id"] + "-" + arm]["assessment"][
                                    "write_boundary_correct"
                                ]
                                and runs[case["id"] + "-" + arm]["parameters_correct"]
                            ),
                            "review_screen_pass": runs[case["id"] + "-" + arm][
                                "review_screen_pass"
                            ],
                            "critical_issues": runs[case["id"] + "-" + arm][
                                "critical_issues"
                            ],
                            "tokens": runs[case["id"] + "-" + arm]["usage"][
                                "reported_tokens"
                            ],
                            "retrieval_calls": runs[case["id"] + "-" + arm]["assessment"][
                                "search_count"
                            ],
                            "answer_citation_ids": [
                                x["chunk_id"]
                                for x in runs[case["id"] + "-" + arm]["first_sources"]
                            ],
                        }
                        for arm in ("full", "tool_rag")
                    },
                }
                for case in cases
            ],
        }
        write(OUT / "summary.json", summary)
        print(json.dumps(summary["comparison"], ensure_ascii=False, indent=2))
        print(json.dumps(summary["by_split"], ensure_ascii=False, indent=2))
    finally:
        index.close()


if __name__ == "__main__":
    main()
