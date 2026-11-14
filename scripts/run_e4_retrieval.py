"""Real local L0/D0/H0/H1 retrieval. Default dev only; held-out requires a frozen protocol."""

import argparse
import asyncio
import json
import platform
import shutil
import statistics
import time
import uuid
from pathlib import Path

from aftersalesagent.evaluation.e4.execution import source_fingerprints
from aftersalesagent.evaluation.e4.scoring import retrieval_metrics, summarize
from aftersalesagent.evaluation.e4.suite import load
from aftersalesagent.knowledge.corpus import dump, load_corpus
from aftersalesagent.knowledge.e4 import E4Config, E4Index, LocalModelClient, vector_signature

ROOT = Path(__file__).resolve().parents[1]


def percentile(values, p):
    return sorted(values)[min(len(values) - 1, int((len(values) - 1) * p))] if values else None


async def run(args):
    if args.split != "dev":
        raise ValueError(
            "Use frozen paired runner for held-out evaluation; no free candidate sweep"
        )
    cases, manifest = load(ROOT, args.split)
    chunks, _ = load_corpus(ROOT, "e4_v1")
    base = E4Config.model_validate_json(
        (ROOT / "configs/evaluation/e4_dense_v1.json").read_text("utf-8")
    )
    cache = json.loads((ROOT / base.vector_cache).read_text("utf-8"))
    if cache["signature"] != vector_signature(chunks, base):
        raise ValueError("Stale E4 vectors")
    directory = ROOT / ".local/e4_runs" / ("retrieval_" + uuid.uuid4().hex[:12])
    directory.mkdir(parents=True, exist_ok=False)
    model = LocalModelClient(ROOT, base)
    result = {
        "run_id": directory.name,
        "split": args.split,
        "real_local_models": True,
        "generation_api_calls": 0,
        "environment": platform.platform(),
        "suite_manifest": manifest,
        "config": base.model_dump(),
        "methods": {},
        "source_hashes": source_fingerprints(ROOT),
    }
    (directory / "config.json").write_bytes(dump(result))
    for relative in result["source_hashes"]:
        target = directory / "snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    try:
        startup = time.perf_counter()
        result["model_info"] = model.request("info")
        result["model_startup_seconds"] = time.perf_counter() - startup
        for label, method in (("L0", "bm25"), ("D0", "dense"), ("H0", "hybrid"), ("H1", "rerank")):
            index = E4Index(
                chunks,
                base.model_copy(update={"method": method}),
                model=model,
                vectors=cache["vectors"] if method != "bm25" else None,
            )
            rows = []
            try:
                for case in cases:
                    public, gold = case["public"], case["criteria"]
                    ranked = await asyncio.to_thread(
                        index.ranked, public["question"], public["filters"], 10
                    )
                    ids = [c.chunk_id for c, _ in ranked]
                    row = {
                        "task_id": public["task_id"],
                        "challenge": gold["challenge"],
                        "secondary_tags": gold["secondary_tags"],
                        "family_id": gold["family_id"],
                        "ids": ids,
                        "metrics": retrieval_metrics(ids, gold),
                        "trace": index.last_trace,
                    }
                    rows.append(row)
                    with (directory / (label + ".jsonl")).open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    print(
                        json.dumps(
                            {
                                "method": label,
                                "task": public["task_id"],
                                "complete": row["metrics"]["complete_support"],
                            }
                        ),
                        flush=True,
                    )
            finally:
                index.close()
            times = [r["trace"]["timings"]["retrieval_seconds"] for r in rows]
            challenges = {
                tag: summarize([r for r in rows if r["challenge"] == tag])
                for tag in sorted({r["challenge"] for r in rows})
            }
            result["methods"][label] = {
                "metrics": summarize(rows),
                "challenges": challenges,
                "median_seconds": statistics.median(times),
                "p95_seconds": percentile(times, 0.95),
                "first_query_seconds": times[0],
            }
            (directory / "report.json").write_bytes(dump(result))
        result["model_info_after_matrix"] = model.request("info")
        result["vector_cache_bytes"] = (ROOT / base.vector_cache).stat().st_size
        (directory / "report.json").write_bytes(dump(result))
    finally:
        model.close()
    print(
        json.dumps(
            {"report": str(directory / "report.json"), "methods": result["methods"]},
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "validation", "test"), default="dev")
    asyncio.run(run(parser.parse_args()))
