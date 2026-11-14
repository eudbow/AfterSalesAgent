"""Real paired DeepSeek E4 answers, global 300-call ledger, no retry and frozen hold-out gates."""

import argparse
import asyncio
import json
import sqlite3
import uuid
from pathlib import Path

from aftersalesagent.business.database import RetailStore
from aftersalesagent.business.service import RetailService
from aftersalesagent.evaluation.e4.budget import GenerationBudget
from aftersalesagent.evaluation.e4.execution import BudgetedBackend, sha, verify_fingerprints
from aftersalesagent.evaluation.e4.scoring import answer_checks, retrieval_metrics
from aftersalesagent.evaluation.e4.suite import load
from aftersalesagent.knowledge.corpus import dump
from aftersalesagent.knowledge.e4 import E4Config, build_index
from aftersalesagent.knowledge.service import KnowledgeQuestion, KnowledgeService
from aftersalesagent.local_config import load_local_config
from aftersalesagent.models.deepseek import DeepSeekBackend, DeepSeekOptions
from aftersalesagent.policies.application import load_application_policy

ROOT = Path(__file__).resolve().parents[1]


async def run(args):
    if not args.enable_paid_api:
        raise ValueError(
            "Real E4 answers require --enable-paid-api; user already authorized up to 300"
        )
    protocol_path = ROOT / "configs/evaluation/e4_protocol_v1.json"
    protocol = json.loads(protocol_path.read_text("utf-8"))
    verify_fingerprints(ROOT, protocol["files"])
    if args.split != "dev":
        previous = "dev" if args.split == "validation" else "validation"
        gate_path = ROOT / f".local/e4_artifacts/{previous}_gate.json"
        gate = json.loads(gate_path.read_text("utf-8"))
        if not gate["passed"] or gate["protocol_sha256"] != sha(protocol_path):
            raise ValueError("Previous frozen E4 stage gate not passed")
    cases, _ = load(ROOT, args.split, protocol["suite_version"])
    if args.split == "dev":
        order = {task_id: n for n, task_id in enumerate(protocol["smoke_task_ids"])}
        cases.sort(
            key=lambda case: (order.get(case["public"]["task_id"], 100), case["public"]["task_id"])
        )
    directory = (
        args.run_dir or ROOT / ".local/e4_runs" / (args.split + "_answers_" + uuid.uuid4().hex[:12])
    ).resolve()
    if not directory.is_relative_to(ROOT / ".local/e4_runs"):
        raise ValueError("E4 run directory must stay in project run area")
    directory.mkdir(parents=True, exist_ok=True)
    run_config = {
        "split": args.split,
        "suite_version": protocol["suite_version"],
        "protocol_sha256": sha(protocol_path),
        "real_api": True,
        "backend": "deepseek",
        "selected": protocol["selected"],
        "options": protocol["options"],
    }
    if (directory / "config.json").exists():
        if json.loads((directory / "config.json").read_text("utf-8")) != run_config:
            raise ValueError("Run resume configuration mismatch")
    else:
        (directory / "config.json").write_bytes(dump(run_config))
    settings, secret = load_local_config(ROOT / ".env")
    options = DeepSeekOptions.model_validate(protocol["options"])
    if settings.model_name != options.model or not secret.get_secret_value():
        raise ValueError("Project model/key configuration mismatch")
    budget = GenerationBudget(
        ROOT / ".local/e4_artifacts/generation_budget.sqlite", protocol["request_limit"]
    )
    backend = BudgetedBackend(
        DeepSeekBackend(options, secret, allow_network=True),
        budget,
        directory.name,
        ROOT / ".local/e4_artifacts/provider_identity_v1.json",
    )
    store = RetailStore(ROOT / ".local/databases/retail_template.sqlite")
    try:
        tools = RetailService(store).specs()
    finally:
        store.close()
    services = {}
    try:
        for label in ("L0", protocol["selected"]):
            method = protocol["methods"][label]
            config_path = ROOT / f"configs/evaluation/e4_{method}_v1.json"
            config = E4Config.model_validate_json(config_path.read_text("utf-8"))
            index = build_index(ROOT, config)
            services[label] = KnowledgeService(
                index,
                backend,
                core_policy=load_application_policy(ROOT),
                tools=tools,
                model_name=options.model,
                temperature=options.temperature,
                max_output_tokens=options.max_output_tokens,
                top_k=protocol["top_k"],
                backend_label="deepseek",
                signature=run_config["protocol_sha256"] + label,
                timeout_seconds=options.timeout_seconds + 5,
                ledger=sqlite3.connect(directory / (label + ".sqlite"), check_same_thread=False),
                close_ledger=True,
            )
        rows = []
        for position, case in enumerate(cases):
            public, gold = case["public"], case["criteria"]
            labels = (
                ("L0", protocol["selected"]) if position % 2 == 0 else (protocol["selected"], "L0")
            )
            for label in labels:
                call_key = directory.name + ":" + public["task_id"] + ":" + label
                output = directory / (public["task_id"] + "_" + label + ".json")
                if output.exists():
                    row = json.loads(output.read_text("utf-8"))
                else:
                    backend.call_key = call_key
                    result = await services[label].ask(
                        KnowledgeQuestion(question=public["question"], **public["filters"]),
                        public["task_id"] + "_" + label,
                    )
                    row = {
                        "task_id": public["task_id"],
                        "method": label,
                        "split": args.split,
                        "family_id": gold["family_id"],
                        "challenge": gold["challenge"],
                        "secondary_tags": gold["secondary_tags"],
                        "result": result,
                        "automatic": answer_checks(result, gold),
                        "retrieval_metrics": retrieval_metrics(result["context_ids"], gold),
                    }
                    output.write_bytes(dump(row))
                rows.append(row)
                (directory / "progress.json").write_bytes(
                    dump(
                        {
                            "completed_pairs": len(rows) // 2,
                            "responses": len(rows),
                            "budget": budget.report(),
                        }
                    )
                )
                print(
                    json.dumps(
                        {
                            "task": public["task_id"],
                            "method": label,
                            "automatic_pass": row["automatic"]["automatic_pass"],
                            "usage": row["result"]["usage"],
                            "error": row["result"]["error"],
                        }
                    ),
                    flush=True,
                )
                if row["result"]["error"] or row["result"]["usage"]["unknown_usage_calls"]:
                    raise ValueError("E4 API technical failure/unknown usage; no automatic retry")
            if args.smoke_only and position == 5:
                break
        (directory / "report.json").write_bytes(
            dump(
                {
                    "config": run_config,
                    "responses": len(rows),
                    "complete_split": len(rows) == 2 * len(cases),
                    "automatic_passes": sum(r["automatic"]["automatic_pass"] for r in rows),
                    "semantic_review": "needs_review",
                    "budget": budget.report(),
                }
            )
        )
    finally:
        for service in services.values():
            await service.close()
        if not services:
            await backend.aclose()
        budget.close()
    print(json.dumps({"run_dir": str(directory), "responses": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--enable-paid-api", action="store_true")
    parser.add_argument("--split", choices=("dev", "validation", "test"), default="dev")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--smoke-only", action="store_true")
    asyncio.run(run(parser.parse_args()))
