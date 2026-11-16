"""Independent v2 batches over the unchanged 24 cases, with unlimited total attempts."""

import argparse
import asyncio
import hashlib
import json

import run_retail_rag_comparison as legacy

from aftersalesagent.knowledge.retail_v2 import RetailV2Index
from aftersalesagent.local_config import load_local_config, options_from_settings
from aftersalesagent.policies.application import load_retail_core_policy
from aftersalesagent.policies.retail_capabilities import RETAIL_APP_CAPABILITIES
from aftersalesagent.schemas.contracts import RunLimits

BASE = legacy.ROOT / "data/results/retail_agent_rag_v2"


class AccountedBackend:
    def __init__(self, backend, run_id):
        self.backend, self.run_id = backend, run_id

    async def generate(self, request):
        ledger = legacy.read_json(legacy.LEDGER)
        run = ledger["runs"][-1]
        if run["id"] != self.run_id or run["actual_attempts"] >= 12:
            raise RuntimeError("Per-session API attempts reached or ledger mismatch")
        run["actual_attempts"] += 1
        ledger["reserved_api_attempts"] += 1
        legacy.write_json(legacy.LEDGER, ledger)
        try:
            response = await self.backend.generate(request)
            ledger = legacy.read_json(legacy.LEDGER)
            run = ledger["runs"][-1]
            run["input_tokens"] += response.input_tokens or 0
            run["output_tokens"] += response.output_tokens or 0
            run["unknown_usage_calls"] += int(response.input_tokens is None)
            legacy.write_json(legacy.LEDGER, ledger)
            return response
        finally:
            ledger = legacy.read_json(legacy.LEDGER)
            ledger["reserved_api_attempts"] -= 1
            ledger["completed_api_attempts"] += 1
            legacy.write_json(legacy.LEDGER, ledger)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=legacy.Path, required=True)
    parser.add_argument("--enable-paid-api", action="store_true")
    parser.add_argument("--batch", required=True)
    parser.add_argument("--case", action="append")
    args = parser.parse_args()
    if not args.enable_paid_api:
        parser.error("Explicit paid API flag required")
    if not args.batch.replace("-", "").replace("_", "").isalnum():
        parser.error("Batch must be a simple name")
    legacy.OUT = BASE / args.batch
    legacy.LEDGER = legacy.OUT / "api_ledger.json"
    if legacy.OUT.exists():
        parser.error("Batch exists; choose a new name to preserve all results")
    settings, secret = load_local_config(args.env_file)
    options = options_from_settings(settings)
    if not secret.get_secret_value():
        parser.error("Local API key missing")
    frozen = legacy.read_json(legacy.CASES)
    digest = hashlib.sha256(
        json.dumps(
            {k: v for k, v in frozen.items() if k != "sha256"},
            sort_keys=True,
            ensure_ascii=False,
        ).encode()
    ).hexdigest()
    if digest != frozen["sha256"]:
        raise ValueError("Original cases changed")
    cases = [c for c in frozen["cases"] if not args.case or c["id"] in args.case]
    if args.case and {c["id"] for c in cases} != set(args.case):
        parser.error("Unknown case")
    index = RetailV2Index()
    legacy.AccountedBackend = AccountedBackend
    policy = load_retail_core_policy() + RETAIL_APP_CAPABILITIES
    legacy.policy_for_arm = lambda _arm, _evidence: policy
    try:
        legacy.write_json(
            legacy.OUT / "protocol.json",
            {
                "scenario_sha256": digest,
                "method": index.method,
                "fingerprint": index.fingerprint,
                "model": options.model,
                "policy_sha256": hashlib.sha256(policy.encode()).hexdigest(),
                "cases": [c["id"] for c in cases],
                "review": "pending",
                "source_hashes": {
                    str(p.relative_to(legacy.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [
                        legacy.ROOT / f
                        for f in (
                            "src/aftersalesagent/knowledge/retail_v2.py",
                            "src/aftersalesagent/policies/retail_v2.py",
                            "src/aftersalesagent/runtime/langgraph_runtime.py",
                            "scripts/run_retail_rag_v2.py",
                        )
                    ]
                },
            },
        )
        legacy.write_json(
            legacy.LEDGER,
            {
                "actual_trajectories": 0,
                "completed_api_attempts": 0,
                "reserved_api_attempts": 0,
                "runs": [],
                "cumulative_limit": None,
            },
        )
        for case in cases:
            run_id = case["id"] + "-tool_rag"
            ledger = legacy.read_json(legacy.LEDGER)
            ledger["actual_trajectories"] += 1
            ledger["runs"].append(
                {
                    "id": run_id,
                    "status": "running",
                    "actual_attempts": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "unknown_usage_calls": 0,
                }
            )
            legacy.write_json(legacy.LEDGER, ledger)
            result = await legacy.run_case(
                case,
                "tool_rag",
                options,
                secret,
                index,
                limits=RunLimits(
                    max_model_calls=12,
                    max_tool_calls=20,
                    max_active_seconds=180,
                    max_tokens=64000,
                    max_turn_model_calls=8,
                    max_turn_tool_calls=12,
                    max_turn_active_seconds=150,
                    max_turn_tokens=48000,
                ),
            )
            legacy.write_json(legacy.OUT / (run_id + ".json"), result)
            ledger = legacy.read_json(legacy.LEDGER)
            ledger["runs"][-1]["status"] = "completed"
            legacy.write_json(legacy.LEDGER, ledger)
            print(
                json.dumps(
                    {
                        "run": run_id,
                        "status": result["first_status"],
                        "assessment": result["assessment"],
                    }
                ),
                flush=True,
            )
    finally:
        index.close()


if __name__ == "__main__":
    asyncio.run(main())
