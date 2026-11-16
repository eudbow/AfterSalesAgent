"""Explicit CUDA-only BGE embedding build; offline pinned files, no model API."""

import hashlib
import json
import os
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no silent CPU fallback")
    started = time.perf_counter()
    lock = json.loads((ROOT / ".local/e4_artifacts/bge-small-zh-v1.5.json").read_text("utf-8"))
    model_path = ROOT / lock["local_path"]
    for record in lock["files"]:
        if digest((model_path / record["name"]).read_bytes()) != record["sha256"]:
            raise ValueError("Pinned model file mismatch: " + record["name"])
    corpus_dir = ROOT / "data/knowledge/retail_sim_v1"
    manifest = json.loads((corpus_dir / "manifest.json").read_text("utf-8"))
    for rel, expected in manifest["files"].items():
        source = (ROOT / rel).resolve()
        if not source.is_relative_to(ROOT) or digest(source.read_bytes()) != expected:
            raise ValueError("Corpus source hash mismatch: " + rel)
    chunks = [
        json.loads(line) for line in (corpus_dir / "chunks.jsonl").read_text("utf-8").splitlines()
    ]
    model = SentenceTransformer(str(model_path), device="cuda", local_files_only=True)
    model.max_seq_length = 512
    model.eval()
    # Small overlapping retrieval units retain parent identity; answers use full parent cards.
    units = []
    for chunk in chunks:
        for start in range(0, len(chunk["text"]), 160):
            end = min(start + 200, len(chunk["text"]))
            text = chunk["section"] + "\n" + chunk["text"][start:end]
            length = len(model.tokenizer.encode(text, truncation=False))
            if length > 512:
                raise ValueError("Embedding window exceeds token budget")
            units.append(
                dict(
                    chunk_id=chunk["chunk_id"], start=start, end=end, text=text, token_length=length
                )
            )
            if end == len(chunk["text"]):
                break
    vectors = model.encode(
        [u["text"] for u in units], batch_size=8, normalize_embeddings=True, show_progress_bar=False
    )
    instruction = "为这个句子生成表示以用于检索相关文章："
    queries = ["原卡不用了可以退到另一张银行卡吗", "退货多久到账", "已处理订单能取消吗"]
    query_vectors = model.encode(
        [instruction + q for q in queries],
        batch_size=3,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    if vectors.shape != (len(units), 512) or not np.isfinite(vectors).all():
        raise ValueError("Invalid vectors")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=0.01):
        raise ValueError("Vectors are not normalized")
    ranking = []
    for query, vector in zip(queries, query_vectors, strict=True):
        scores = vectors @ vector
        parents = {}
        for unit, score in zip(units, scores, strict=True):
            parents[unit["chunk_id"]] = max(parents.get(unit["chunk_id"], -1), float(score))
        ranking.append(dict(query=query, top5=sorted(parents.items(), key=lambda x: -x[1])[:5]))
    out = ROOT / "data/results/retail_agent_rag_v1/embedding"
    out.mkdir(parents=True, exist_ok=True)
    cache = dict(units=units, vectors=vectors.tolist())
    raw = json.dumps(cache, ensure_ascii=False, allow_nan=False).encode("utf-8")
    (out / "vectors.json").write_bytes(raw)
    report = dict(
        model=lock["model_id"],
        revision=lock["revision"],
        device=str(model.device),
        gpu=torch.cuda.get_device_name(0),
        precision="fp32",
        torch=torch.__version__,
        corpus_manifest_sha256=digest(json.dumps(manifest, sort_keys=True).encode()),
        vector_content_sha256=digest(raw),
        instruction=instruction,
        normalization="l2",
        dimension=512,
        max_length=512,
        window_characters=200,
        stride_characters=160,
        parent_aggregation="max",
        chunks=len(chunks),
        units=len(units),
        max_observed_tokens=max(u["token_length"] for u in units),
        elapsed_seconds=time.perf_counter() - started,
        cuda_peak_bytes=torch.cuda.max_memory_allocated(),
        smoke=ranking,
        limitation="Development embedding smoke, not end-to-end or held-out evaluation",
    )
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
