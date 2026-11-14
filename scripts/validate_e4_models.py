"""Explicit real-model CPU/GPU checks and local vector cache creation; never a pytest hook."""

import json
import subprocess
import time
from pathlib import Path

from aftersalesagent.knowledge.corpus import dump, load_corpus
from aftersalesagent.knowledge.e4 import E4Config, LocalModelClient, check_vectors, vector_signature

ROOT = Path(__file__).resolve().parents[1]


def main():
    artifact = ROOT / ".local/e4_artifacts"
    previous = artifact / "model_validation.json"
    if previous.exists():
        history = artifact / "model_validations"
        history.mkdir(exist_ok=True)
        (history / (str(time.time_ns()) + ".json")).write_bytes(previous.read_bytes())
    lock = json.loads((artifact / "bge-small-zh-v1.5.json").read_text("utf-8"))
    reranker = json.loads((artifact / "bge-reranker-v2-m3.json").read_text("utf-8"))
    report = {"real_model": True, "remote_generation_calls": 0, "checks": {}}
    probes = ["SKYTTA 地面轨道胶带拆除损伤地板", "宜家滑门胶条撕掉后地面留痕，是否质保？"]
    config = E4Config(
        method="dense", embedding_revision=lock["revision"], reranker_revision=reranker["revision"]
    )
    cpu = LocalModelClient(ROOT, config)
    try:
        start = time.perf_counter()
        report["cpu_info"] = cpu.request("info")
        report["cpu_startup_seconds"] = time.perf_counter() - start
        result = cpu.request("encode", texts=probes, kind="query")
        vectors = result["vectors"]
        check_vectors(vectors, 2)
        report["cpu_encoding"] = {k: v for k, v in result.items() if k != "vectors"}
        report["checks"]["cpu_512_normalized"] = True
        ranking = cpu.request(
            "rerank",
            pairs=[
                [probes[0], "本品质保证不涵盖取下胶带后地板表面留下的印记或损坏。"],
                [probes[0], "无不粘涂层的宜家锅具可享受15年免费质保。"],
            ],
        )
        report["cpu_reranking"] = ranking
        report["checks"]["reranker_relevant_above_unrelated"] = (
            ranking["scores"][0] > ranking["scores"][1]
        )
    finally:
        cpu.close()
    config = config.model_copy(update={"device": "cuda", "precision": "fp16"})
    gpu = LocalModelClient(ROOT, config)
    try:
        start = time.perf_counter()
        report["gpu_info"] = gpu.request("info")
        report["gpu_startup_seconds"] = time.perf_counter() - start
        result = gpu.request("encode", texts=probes, kind="query")
        check_vectors(result["vectors"], 2)
        similarities = [
            sum(a * b for a, b in zip(x, y, strict=True))
            for x, y in zip(vectors, result["vectors"], strict=True)
        ]
        report["gpu_encoding"] = {k: v for k, v in result.items() if k != "vectors"}
        report["cpu_gpu_cosines"] = similarities
        report["checks"]["gpu_512_normalized"] = True
        report["checks"]["cpu_gpu_cosine_above_099"] = all(x > 0.99 for x in similarities)
        chunks, _ = load_corpus(ROOT, "e4_v1")
        text = [c.section + "\n" + c.text for c in chunks]
        result = gpu.request("encode", texts=text, kind="document", batch_size=2)
        check_vectors(result["vectors"], len(chunks))
        cache = {
            "signature": vector_signature(chunks, config),
            "chunk_ids": [c.chunk_id for c in chunks],
            "vectors": result["vectors"],
            "encoding_seconds": result["elapsed_seconds"],
            "lengths": result["lengths"],
            "environment": report["gpu_info"],
        }
        (artifact / "vectors.json").write_bytes(dump(cache))
        report["corpus_encoding"] = {k: v for k, v in result.items() if k != "vectors"}
        report["checks"]["no_silent_truncation"] = max(result["lengths"]) <= 512
        config_dir = ROOT / "configs/evaluation"
        for method in ("bm25", "dense", "hybrid", "rerank"):
            selected = config.model_copy(update={"method": method})
            (config_dir / f"e4_{method}_v1.json").write_bytes(dump(selected.model_dump()))
    finally:
        gpu.close()
    report["passed"] = all(report["checks"].values())
    (artifact / "model_validation.json").write_bytes(dump(report))
    python = ROOT / config.worker_python
    result = subprocess.run(
        [str(python), "-m", "pip", "freeze"], capture_output=True, text=True, check=True
    )
    (artifact / "model-requirements.lock.txt").write_text(result.stdout, "utf-8")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "checks": report["checks"],
                "gpu_peak_bytes": report["gpu_encoding"]["cuda_peak_bytes"],
                "corpus_max_tokens": max(report["corpus_encoding"]["lengths"]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
