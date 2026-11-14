"""Explicit C3 vector preparation, preflight/freeze, real three-arm runs and review gates."""

import argparse
import asyncio
import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

from aftersalesagent.evaluation.e4.budget import GenerationBudget
from aftersalesagent.evaluation.e4.execution import (
    BudgetedBackend,
    sha,
    source_fingerprints,
    verify_fingerprints,
)
from aftersalesagent.evaluation.e4.scoring import retrieval_metrics
from aftersalesagent.evaluation.e4_c3 import (
    ART,
    GUIDANCE,
    OPTIONS,
    PROTOCOL_PATH,
    REPEAT_IDS,
    RUN,
    SMOKE_IDS,
    C3Config,
    automatic,
    call_plan,
    evaluate_review,
    indexes,
    input_size,
    load_materials,
    make_request,
    prepare_reviews,
    stage_allowed,
    summarize_rows,
    tool_specs,
    worker_config,
    write_new,
)
from aftersalesagent.knowledge.corpus import dump, eligible
from aftersalesagent.knowledge.e4 import LocalModelClient, check_vectors, vector_signature
from aftersalesagent.knowledge.service import Answer
from aftersalesagent.local_config import load_local_config
from aftersalesagent.models.deepseek import DeepSeekBackend, DeepSeekOptions
from aftersalesagent.models.errors import ModelBackendFault

ROOT = Path(__file__).resolve().parents[1]


def prepare_vectors():
    config = C3Config()
    if (ROOT / config.vector_cache).exists():
        raise ValueError("C3 vectors already exist")
    chunks, _, _ = load_materials(ROOT, "dev")
    model = LocalModelClient(ROOT, worker_config(config))
    try:
        info = model.request("info")
        output = model.request(
            "encode",
            texts=[c.section + "\n" + c.text for c in chunks],
            kind="document",
            batch_size=2,
        )
        check_vectors(output["vectors"], len(chunks))
        write_new(
            ROOT / config.vector_cache,
            {
                "signature": vector_signature(chunks, worker_config(config)),
                "chunk_ids": [c.chunk_id for c in chunks],
                "vectors": output["vectors"],
                "lengths": output["lengths"],
                "elapsed_seconds": output["elapsed_seconds"],
                "environment": info,
            },
        )
        print(
            json.dumps(
                {
                    "chunks": len(chunks),
                    "max_tokens": max(output["lengths"]),
                    "real_local_model": True,
                    "generation_calls": 0,
                }
            )
        )
    finally:
        model.close()


def preflight():
    tools = tool_specs(ROOT)
    sizes = []
    for split in ("dev", "validation", "test"):
        chunks, cases, _ = load_materials(ROOT, split)
        for case in cases:
            p = case["public"]
            context = eligible(chunks, p["filters"])
            size = input_size(make_request(ROOT, p, context, tools, OPTIONS), OPTIONS)
            sizes.append(
                {"split": split, "task_id": p["task_id"], "eligible_chunks": len(context), **size}
            )
    maximum = max(r["serialized_characters"] for r in sizes)
    byte_max = max(r["serialized_utf8_bytes"] for r in sizes)
    if maximum > 200000 or byte_max > 500000:
        raise ValueError("Offline full-input safety bound exceeded; no truncation")
    report = {
        "full_inputs": sizes,
        "max_characters": maximum,
        "max_utf8_bytes": byte_max,
        "common_character_guard": 200000,
        "truncation": False,
        "provider_documented_context": "1M tokens; official docs checked 2026-11-12",
        "provider_docs": "https://api-docs.deepseek.com/quick_start/pricing/",
        "token_measurement": (
            "Exact provider usage only after each call; no bge token count presented as "
            "DeepSeek input tokens"
        ),
    }
    write_new(ROOT / ART / "preflight_v1.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "full_inputs"}, ensure_ascii=False))


def freeze():
    if (ROOT / PROTOCOL_PATH).exists():
        raise ValueError("C3 protocol already frozen")
    files = source_fingerprints(ROOT)
    materials = json.loads((ROOT / "data/manifests/e4_c3_v1.json").read_text("utf-8"))
    files.update(materials["files"])
    for p in ("data/manifests/e4_c3_v1.json", C3Config().vector_cache, ART + "/preflight_v1.json"):
        files[p] = sha(ROOT / p)
    old = json.loads((ROOT / "configs/evaluation/e4_protocol_v1.json").read_text("utf-8"))
    for p, digest in old["files"].items():
        if p.startswith(".local/models/") or p in (
            ".local/e4_artifacts/model-requirements.lock.txt",
            ".local/e4_artifacts/bge-small-zh-v1.5.json",
        ):
            if sha(ROOT / p) != digest:
                raise ValueError("Pinned model/environment drift: " + p)
            files[p] = digest
    spec = {
        "version": "e4_c3_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "methods": ["Full", "BM25", "Hybrid"],
        "retrieval": C3Config().model_dump(),
        "options": OPTIONS,
        "request_limit": 300,
        "main_calls": 270,
        "repeated_calls": 30,
        "repeat_ids": REPEAT_IDS,
        "smoke_ids": SMOKE_IDS,
        "character_guard": 200000,
        "prompt": GUIDANCE,
        "files": files,
        "validity_gate": (
            "complete known usage, no technical failure, every semantic review committed, "
            "no unresolved review; no quality-win condition"
        ),
        "quality_decision": (
            "Report paired differences and family counts; no noninferiority/significance "
            "claim from equal small-sample counts; keep BM25 unless justified"
        ),
        "reviewer_kind": "assistant",
        "price_estimate": None,
    }
    write_new(ROOT / PROTOCOL_PATH, spec)
    snapshot = ROOT / RUN / "execution_snapshot"
    snapshot.mkdir(parents=True, exist_ok=False)
    for rel in files:
        if rel.startswith(".local/models/") or rel.endswith("vectors_v1.json"):
            continue
        dest = snapshot / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dest)
    shutil.copy2(ROOT / PROTOCOL_PATH, snapshot / "protocol.json")
    print(
        json.dumps(
            {
                "protocol": PROTOCOL_PATH,
                "sha256": sha(ROOT / PROTOCOL_PATH),
                "files": len(files),
                "request_cap": 300,
            }
        )
    )


async def run(args):
    if not args.execute or not args.enable_paid_api:
        raise ValueError("Explicit --execute --enable-paid-api required")
    spec = json.loads((ROOT / PROTOCOL_PATH).read_text("utf-8"))
    digest = sha(ROOT / PROTOCOL_PATH)
    verify_fingerprints(ROOT, spec["files"])
    stage_allowed(ROOT, args.split, digest)
    chunks, cases, _ = load_materials(ROOT, args.split)
    plan = call_plan(cases, args.split, args.smoke_only)
    tools = tool_specs(ROOT)
    settings, secret = load_local_config(ROOT / ".env")
    if settings.model_name != spec["options"]["model"] or not secret.get_secret_value():
        raise ValueError("Project model/key configuration mismatch")
    budget = GenerationBudget(ROOT / ART / "generation_budget.sqlite", 300)
    backend = BudgetedBackend(
        DeepSeekBackend(
            DeepSeekOptions.model_validate(spec["options"]), secret, allow_network=True
        ),
        budget,
        "e4_c3_main_v1",
        ROOT / ART / "provider_identity_v1.json",
    )
    engines = indexes(ROOT, chunks)
    directory = ROOT / RUN / args.split
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        for case, method, rep in plan:
            p = case["public"]
            gold = case["criteria"]
            name = f"{p['task_id']}_{method}_r{rep}"
            target = directory / (name + ".json")
            if target.exists():
                row = json.loads(target.read_text("utf-8"))
                if row["protocol_sha256"] != digest:
                    raise ValueError("Resume mismatch")
                rows.append(row)
                continue
            started = time.perf_counter()
            allowed = eligible(chunks, p["filters"])
            if method == "Full":
                context = allowed
                trace = {"method": "full", "selected_ids": [c.chunk_id for c in context]}
                ranking = []
            else:
                ranked = await asyncio.to_thread(
                    engines[method].ranked, p["question"], p["filters"], 10
                )
                ranking = [c.chunk_id for c, s in ranked]
                context = [c for c, s in ranked[:5]]
                trace = engines[method].last_trace
            retrieval_seconds = time.perf_counter() - started
            request = make_request(ROOT, p, context, tools, spec["options"])
            size = input_size(request, spec["options"])
            if size["serialized_characters"] > spec["character_guard"]:
                raise ValueError("Input exceeds frozen guard; no silent truncation")
            backend.call_key = "c3:" + args.split + ":" + name
            response = None
            error = None
            answer = None
            generation_started = time.perf_counter()
            try:
                response = await backend.generate(request)
                if response.tool_calls or response.finish_reason != "stop":
                    raise ValueError("Unexpected model tool/finish")
                answer = Answer.model_validate_json(response.content).model_dump()
            except ModelBackendFault as exc:
                error = exc.code
                response = exc.observation
            except Exception as exc:
                error = type(exc).__name__
            generation_seconds = time.perf_counter() - generation_started
            by_id = {c.chunk_id: c for c in context}
            ids = answer["citations"] if answer else []
            valid_ids = answer is not None and len(ids) == len(set(ids)) and set(ids) <= set(by_id)
            valid_versions = valid_ids and set(answer["policy_versions"]) == {
                by_id[cid].version for cid in ids
            }
            unknown = (
                response is None or response.input_tokens is None or response.output_tokens is None
            )
            result = {
                "question": p,
                "answer": answer,
                "error": error,
                "context_ids": [c.chunk_id for c in context],
                "eligible_ids": [c.chunk_id for c in allowed],
                "sources": [by_id[cid].model_dump() for cid in ids if cid in by_id],
                "citation_ids_valid": valid_ids,
                "citation_versions_valid": valid_versions,
                "model_request": request.model_dump(mode="json"),
                "model_response": response.model_dump(mode="json") if response else None,
                "input_size": size,
                "timings": {
                    "retrieval_seconds": retrieval_seconds,
                    "generation_seconds": generation_seconds,
                    "end_to_end_seconds": time.perf_counter() - started,
                },
                "usage": {
                    "input_tokens": response.input_tokens if response else None,
                    "output_tokens": response.output_tokens if response else None,
                    "unknown_usage_calls": int(unknown),
                },
                "retrieval_trace": trace,
                "cost_estimate": None,
            }
            row = {
                "protocol_sha256": digest,
                "task_id": p["task_id"],
                "family_id": gold["family_id"],
                "method": method,
                "split": args.split,
                "challenge": gold["challenge"],
                "repetition": rep,
                "result": result,
                "automatic": automatic(result, gold),
                "retrieval_metrics": retrieval_metrics(ranking, gold) if method != "Full" else None,
            }
            write_new(target, row)
            rows.append(row)
            (directory / "progress.json").write_bytes(
                dump({"responses_in_plan": len(rows), "budget": budget.report()})
            )
            print(
                json.dumps(
                    {
                        "task": p["task_id"],
                        "method": method,
                        "rep": rep,
                        "automatic": row["automatic"]["automatic_pass"],
                        "usage": result["usage"],
                        "error": error,
                    }
                ),
                flush=True,
            )
            if error or unknown:
                raise ValueError("C3 technical failure/unknown usage; stopped without retry")
        report = {
            "split": args.split,
            "protocol_sha256": digest,
            "responses": len(rows),
            "complete": len(rows) == (120 if args.split == "dev" else 90),
            "real_api": True,
            "summary": summarize_rows(rows),
            "budget": budget.report(),
            "semantic_review": "pending",
        }
        (directory / "report.json").write_bytes(dump(report))
    finally:
        for index in engines.values():
            index.close()
        await backend.aclose()
        budget.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=("vectors", "preflight", "freeze", "run", "prepare-review", "gate")
    )
    parser.add_argument("--split", choices=("dev", "validation", "test"), default="dev")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--enable-paid-api", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    if args.action == "vectors":
        prepare_vectors()
    elif args.action == "preflight":
        preflight()
    elif args.action == "freeze":
        freeze()
    elif args.action == "prepare-review":
        print(json.dumps({"review_items": prepare_reviews(ROOT, args.split)}))
    elif args.action == "gate":
        print(json.dumps(evaluate_review(ROOT, args.split), ensure_ascii=False))
    else:
        asyncio.run(run(args))
