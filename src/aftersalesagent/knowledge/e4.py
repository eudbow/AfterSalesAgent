"""Versioned E4 retrieval with an independent model process and no torch app dependency."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from aftersalesagent.knowledge.base import EvidenceChunk
from aftersalesagent.knowledge.corpus import eligible, load_corpus
from aftersalesagent.knowledge.retrieval import SQLiteKnowledgeIndex


def fingerprint(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


class E4Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    corpus_version: Literal["e4_v1"] = "e4_v1"
    method: Literal["bm25", "dense", "hybrid", "rerank"] = "bm25"
    model_id: Literal["BAAI/bge-small-zh-v1.5"] = "BAAI/bge-small-zh-v1.5"
    embedding_revision: str = ""
    device: Literal["cpu", "cuda"] = "cpu"
    precision: Literal["fp32", "fp16"] = "fp32"
    candidate_k: int = Field(default=20, ge=1, le=30)
    fusion_k: int = Field(default=30, ge=1, le=60)
    rrf_constant: int = Field(default=60, ge=1)
    top_k: int = Field(default=5, ge=1, le=30)
    reranker_revision: str = ""
    worker_python: str = ".local/conda-envs/aftersales-rag/python.exe"
    vector_cache: str = ".local/e4_artifacts/vectors.json"
    worker_timeout_seconds: float = Field(default=180, gt=0, le=600)


class LocalModelClient:
    """One persistent child, serialized requests, bounded waits, no model API key inheritance."""

    def __init__(self, root: Path, config: E4Config):
        self.root, self.config = root.resolve(), config
        python = (self.root / config.worker_python).resolve()
        if not python.is_relative_to(self.root) or not python.is_file():
            raise ValueError("Explicit project-local worker Python is unavailable")
        env = {
            k: os.environ[k]
            for k in ("PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP")
            if k in os.environ
        }
        env.update(
            {
                "PYTHONUTF8": "1",
                "PYTHONIOENCODING": "utf-8",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_HOME": str(self.root / ".local/huggingface"),
            }
        )
        self.process = subprocess.Popen(
            [
                str(python),
                str(self.root / "scripts/e4_model_worker.py"),
                "--root",
                str(self.root),
                "--device",
                config.device,
                "--precision",
                config.precision,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env=env,
            cwd=self.root,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.lock = threading.Lock()
        self.stderr_tail = []
        self.stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self.stderr_thread.start()

    def _drain_stderr(self):
        if self.process.stderr:
            for line in self.process.stderr:
                self.stderr_tail = (self.stderr_tail + [line])[-12:]

    def request(self, method: str, **payload):
        with self.lock:
            if self.process.poll() is not None:
                raise RuntimeError(
                    "Local model worker exited: " + "".join(self.stderr_tail)[-1500:]
                )
            assert self.process.stdin and self.process.stdout
            self.process.stdin.write(
                json.dumps({"method": method, **payload}, ensure_ascii=False) + "\n"
            )
            self.process.stdin.flush()
            future = self.pool.submit(self.process.stdout.readline)
            try:
                line = future.result(timeout=self.config.worker_timeout_seconds)
            except FutureTimeout:
                self.close()
                raise TimeoutError("Local model worker timed out; no automatic restart") from None
            if not line:
                raise RuntimeError(
                    "Local model worker closed: " + "".join(self.stderr_tail)[-1500:]
                )
            result = json.loads(line)
            if not result.get("ok"):
                raise ValueError("Local model failed: " + str(result.get("error")))
            return result

    def encode(self, texts, *, kind="document"):
        return self.request("encode", texts=texts, kind=kind)["vectors"]

    def rerank(self, query, chunks):
        return self.request("rerank", pairs=[[query, c.section + "\n" + c.text] for c in chunks])[
            "scores"
        ]

    def close(self):
        if self.process.poll() is None:
            if self.process.stdin:
                self.process.stdin.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.pool.shutdown(wait=False, cancel_futures=True)


def vector_signature(chunks, config: RetrievalConfig):
    return fingerprint(
        {
            "corpus": [(c.chunk_id, c.model_dump()) for c in chunks],
            "model": config.model_id,
            "revision": config.embedding_revision,
            "precision": config.precision,
            "device": config.device,
            "instruction": "为这个句子生成表示以用于检索相关文章：",
            "pooling": "cls",
            "normalization": "l2",
            "max_length": 512,
        }
    )


def check_vectors(vectors, count):
    if len(vectors) != count:
        raise ValueError("Vector count mismatch")
    for row in vectors:
        if len(row) != 512 or any(not math.isfinite(x) for x in row):
            raise ValueError("Embedding dimension or finite-value check failed")
        if abs(sum(x * x for x in row) - 1) > 0.01:
            raise ValueError("Embedding must be L2 normalized")


class RetrievalConfig(Protocol):
    """Read-only shape shared by versioned experiment configurations."""

    @property
    def corpus_version(self) -> str: ...

    @property
    def model_id(self) -> str: ...

    @property
    def embedding_revision(self) -> str: ...

    @property
    def precision(self) -> str: ...

    @property
    def device(self) -> str: ...

    @property
    def method(self) -> str: ...

    @property
    def candidate_k(self) -> int: ...

    @property
    def fusion_k(self) -> int: ...

    @property
    def rrf_constant(self) -> int: ...

    def model_dump(self) -> dict[str, Any]: ...


class E4Index:
    def __init__(
        self, chunks, config: RetrievalConfig, *, model=None, vectors=None, owns_model=False
    ):
        self.chunks, self.config, self.model = chunks, config, model
        self.owns_model = owns_model
        self.tokenizer = "e4-char-" + config.method
        if config.method != "bm25":
            if model is None or vectors is None:
                raise ValueError("Dense paths require explicit model and validated vector cache")
            check_vectors(vectors, len(chunks))
        self.lexical = SQLiteKnowledgeIndex(chunks, "char")
        self.by_id = {c.chunk_id: c for c in chunks}
        self.index_fingerprint = fingerprint(
            {
                "config": config.model_dump(),
                "vectors": vector_signature(chunks, config),
                "vector_content_sha256": fingerprint(vectors) if vectors is not None else None,
            }
        )
        self.vectors = vectors
        self.last_trace = None

    def select(self, filters):
        return eligible(self.chunks, filters)

    def ranked(self, query, filters, top_k):
        if not 1 <= top_k <= 30 or not query.strip():
            raise ValueError("Invalid query or top_k")
        started = time.perf_counter()
        allowed = self.select(filters)
        allowed_ids = {c.chunk_id for c in allowed}
        trace = {
            "method": self.config.method,
            "index_fingerprint": self.index_fingerprint,
            "filters": filters,
            "stages": {},
            "timings": {},
        }
        lexical = (
            self.lexical.ranked(query, filters, max(top_k, self.config.candidate_k))
            if self.config.method != "dense"
            else []
        )
        trace["stages"]["bm25"] = [
            {"id": c.chunk_id, "score": s, "direction": "lower"} for c, s in lexical
        ]
        if self.config.method == "bm25":
            result = lexical[:top_k]
        elif not allowed:
            result = []
        else:
            assert self.model is not None and self.vectors is not None
            before = time.perf_counter()
            query_vector = self.model.encode([query], kind="query")[0]
            check_vectors([query_vector], 1)
            trace["timings"]["query_encoding_seconds"] = time.perf_counter() - before
            before = time.perf_counter()
            dense = []
            for chunk, vector in zip(self.chunks, self.vectors, strict=True):
                if chunk.chunk_id in allowed_ids:
                    dense.append(
                        (chunk, sum(a * b for a, b in zip(query_vector, vector, strict=True)))
                    )
            dense.sort(key=lambda pair: (-pair[1], pair[0].chunk_id))
            trace["timings"]["dense_search_seconds"] = time.perf_counter() - before
            trace["stages"]["dense"] = [
                {"id": c.chunk_id, "score": s, "direction": "higher"}
                for c, s in dense[: self.config.candidate_k]
            ]
            if self.config.method == "dense":
                result = dense[:top_k]
            else:
                before = time.perf_counter()
                scores = {}
                for ranking in (lexical, dense[: self.config.candidate_k]):
                    for rank, (chunk, _) in enumerate(ranking, 1):
                        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1 / (
                            self.config.rrf_constant + rank
                        )
                fused = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[
                    : self.config.fusion_k
                ]
                trace["stages"]["fusion"] = [
                    {"id": cid, "score": score, "direction": "higher"} for cid, score in fused
                ]
                trace["timings"]["fusion_seconds"] = time.perf_counter() - before
                result = [(self.by_id[cid], score) for cid, score in fused]
                if self.config.method == "rerank":
                    before = time.perf_counter()
                    values = self.model.rerank(query, [c for c, _ in result])
                    if len(values) != len(result) or any(not math.isfinite(v) for v in values):
                        raise ValueError("Invalid reranker output")
                    result = sorted(
                        zip([c for c, _ in result], values, strict=True),
                        key=lambda pair: (-pair[1], pair[0].chunk_id),
                    )
                    trace["timings"]["reranking_seconds"] = time.perf_counter() - before
                    trace["stages"]["rerank"] = [
                        {"id": c.chunk_id, "score": s, "direction": "higher"} for c, s in result
                    ]
                result = result[:top_k]
        trace["timings"]["retrieval_seconds"] = time.perf_counter() - started
        trace["selected_ids"] = [c.chunk_id for c, _ in result]
        self.last_trace = trace
        return result

    async def provide(self, query, filters, top_k, *, full=False):
        return (
            self.select(filters)
            if full
            else [c for c, _ in await asyncio.to_thread(self.ranked, query, filters, top_k)]
        )

    async def search(self, query, filters, top_k):
        return [
            EvidenceChunk(
                document_id=c.document_id,
                section=c.section,
                text=c.text,
                version=c.version,
                source=c.source,
                score=score,
                chunk_id=c.chunk_id,
                scope=c.scope,
                origin=c.origin,
                retrieval_method=self.config.method,
                score_direction="lower" if self.config.method == "bm25" else "higher",
                index_fingerprint=self.index_fingerprint,
            )
            for c, score in await asyncio.to_thread(self.ranked, query, filters, top_k)
        ]

    def close(self):
        self.lexical.close()
        if self.owns_model and self.model:
            self.model.close()


def build_index(root: Path, config: E4Config, *, model=None):
    chunks, _ = load_corpus(root, config.corpus_version)
    owned = False
    vectors = None
    if config.method != "bm25":
        lock = json.loads((root / ".local/e4_artifacts/bge-small-zh-v1.5.json").read_text("utf-8"))
        if not lock.get("complete") or config.embedding_revision != lock["revision"]:
            raise ValueError("Embedding snapshot fingerprint mismatch")
        if config.method == "rerank":
            reranker = json.loads(
                (root / ".local/e4_artifacts/bge-reranker-v2-m3.json").read_text("utf-8")
            )
            if not reranker.get("complete") or config.reranker_revision != reranker["revision"]:
                raise ValueError("Reranker snapshot fingerprint mismatch")
        cache_path = (root / config.vector_cache).resolve()
        if not cache_path.is_relative_to(root.resolve()):
            raise ValueError("Vector cache escapes project")
        cache = json.loads(cache_path.read_text("utf-8"))
        if cache["signature"] != vector_signature(chunks, config):
            raise ValueError("Vector cache configuration mismatch")
        if cache["chunk_ids"] != [c.chunk_id for c in chunks]:
            raise ValueError("Vector cache evidence order mismatch")
        vectors = cache["vectors"]
        check_vectors(vectors, len(chunks))
        if model is None:
            model, owned = LocalModelClient(root, config), True
    return E4Index(chunks, config, model=model, vectors=vectors, owns_model=owned)
