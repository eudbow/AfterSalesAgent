"""Isolated C3 experiment mechanics; shared E4 retrieval, new materials and paid ledger."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
import statistics
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from aftersalesagent.business.database import RetailStore
from aftersalesagent.business.service import RetailService
from aftersalesagent.evaluation.e4.execution import sha
from aftersalesagent.evaluation.e4.scoring import answer_checks
from aftersalesagent.knowledge.corpus import KnowledgeChunk, dump
from aftersalesagent.knowledge.e4 import (
    E4Config,
    E4Index,
    LocalModelClient,
    check_vectors,
    vector_signature,
)
from aftersalesagent.knowledge.service import PROTOCOL
from aftersalesagent.models.deepseek import DeepSeekOptions, encode_request
from aftersalesagent.policies.application import load_application_policy
from aftersalesagent.schemas.contracts import ModelMessage, ModelRequest

METHODS = ("Full", "BM25", "Hybrid")
REPEAT_IDS = ("C3-D01-1", "C3-D04-1", "C3-D07-1", "C3-D10-1", "C3-D12-1")
SMOKE_IDS = ("C3-D01-1", "C3-D04-1", "C3-D10-1", "C3-D12-1", "C3-D14-1", "C3-D15-1")
ART = ".local/e4_c3_artifacts"
RUN = ".local/e4_c3_runs/main_v1"
PROTOCOL_PATH = "configs/evaluation/e4_c3_protocol_v1.json"
OPTIONS = {
    "model": "deepseek-flash",
    "base_url": "https://api.deepseek.com",
    "temperature": 0,
    "max_output_tokens": 1024,
    "timeout_seconds": 120,
}
GUIDANCE = (
    PROTOCOL + " Document/section labels provide topic scope, not evidence that all "
    "document-level durations apply to every subsection. Use the actual scoped "
    "clauses. Cite a minimal sufficient set, preserving requested conditions and "
    "exceptions. Do not infer an individual order outcome from a general warranty."
)


class C3Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    corpus_version: Literal["e4_c3_v1"] = "e4_c3_v1"
    method: Literal["bm25", "hybrid"] = "hybrid"
    model_id: str = "BAAI/bge-small-zh-v1.5"
    embedding_revision: str = "7999e1d3359715c523056ef9478215996d62a620"
    device: Literal["cpu", "cuda"] = "cuda"
    precision: Literal["fp32", "fp16"] = "fp16"
    candidate_k: int = 20
    fusion_k: int = 30
    rrf_constant: int = 60
    top_k: int = 5
    reranker_revision: str = ""
    worker_python: str = ".local/conda-envs/aftersales-rag/python.exe"
    vector_cache: str = ".local/e4_c3_artifacts/vectors_v1.json"
    worker_timeout_seconds: float = 180


def write_new(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(dump(value))


def load_materials(root: Path, split: str):
    if split not in ("dev", "validation", "test"):
        raise ValueError("Unknown split")
    manifest = json.loads((root / "data/manifests/e4_c3_v1.json").read_text("utf-8"))
    for rel, digest in manifest["files"].items():
        path = (root / rel).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path) != digest:
            raise ValueError("C3 material mismatch: " + rel)
    chunks = [
        KnowledgeChunk.model_validate_json(line)
        for line in (root / "data/knowledge/e4_c3_v1/chunks.jsonl").read_bytes().splitlines()
    ]
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    for c in chunks:
        path = (root / c.source_text_path).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Source outside project")
        text = path.read_text("utf-8")[c.source_start : c.source_end]
        if hashlib.sha256(text.encode()).hexdigest() != c.source_excerpt_sha256:
            raise ValueError("Source span mismatch")
    public = [
        json.loads(line)
        for line in (root / f"data/tasks/e4_c3_v1/{split}.jsonl").read_text("utf-8").splitlines()
    ]
    private = [
        json.loads(line)
        for line in (root / f"data/tasks/private/e4_c3_v1/{split}.jsonl")
        .read_text("utf-8")
        .splitlines()
    ]
    if {r["task_id"] for r in public} != {r["task_id"] for r in private}:
        raise ValueError("Task mismatch")
    gold = {r["task_id"]: r for r in private}
    return chunks, [{"public": p, "criteria": gold[p["task_id"]]} for p in public], manifest


def worker_config(config: C3Config):
    # E4 worker loads only the pinned encoder; corpus loading is exclusively C3's responsibility.
    return E4Config.model_validate(
        {**config.model_dump(exclude={"corpus_version"}), "corpus_version": "e4_v1"}
    )


def indexes(root: Path, chunks: list[KnowledgeChunk]):
    config = C3Config()
    cache = json.loads((root / config.vector_cache).read_text("utf-8"))
    if cache["signature"] != vector_signature(chunks, worker_config(config)) or cache[
        "chunk_ids"
    ] != [c.chunk_id for c in chunks]:
        raise ValueError("C3 vector cache mismatch")
    check_vectors(cache["vectors"], len(chunks))
    model = LocalModelClient(root, worker_config(config))
    return {
        "BM25": E4Index(chunks, config.model_copy(update={"method": "bm25"})),
        "Hybrid": E4Index(chunks, config, model=model, vectors=cache["vectors"], owns_model=True),
    }


def tool_specs(root: Path):
    store = RetailStore(root / ".local/databases/retail_template.sqlite")
    try:
        return RetailService(store).specs()
    finally:
        store.close()


def make_request(
    root: Path, public: dict, context: list[KnowledgeChunk], tools: list, options: dict
):
    fields = {
        "chunk_id",
        "document_id",
        "section",
        "text",
        "version",
        "source",
        "scope",
        "origin",
        "effective_from",
        "effective_until",
    }
    payload = {
        "question": public["question"],
        **public["filters"],
        "evidence": [c.model_dump(mode="json", include=fields) for c in context],
    }
    return ModelRequest(
        messages=[
            ModelMessage(role="system", content=load_application_policy(root)),
            ModelMessage(role="system", content=GUIDANCE),
            ModelMessage(role="user", content=json.dumps(payload, ensure_ascii=False)),
        ],
        tools=tools,
        model_name=options["model"],
        temperature=options["temperature"],
        max_output_tokens=options["max_output_tokens"],
        tool_choice="none",
    )


def input_size(request, options):
    body = encode_request(request, DeepSeekOptions.model_validate(options))
    raw = json.dumps(body, ensure_ascii=False)
    return {
        "serialized_characters": len(raw),
        "serialized_utf8_bytes": len(raw.encode()),
        "exact_provider_input_tokens": None,
        "note": (
            "Offline serialization size, not a tokenizer-equivalent token count; API "
            "usage is authoritative"
        ),
    }


def automatic(result, gold):
    return answer_checks(result, gold)


def stage_allowed(root: Path, split: str, protocol_sha: str):
    if split == "dev":
        return
    previous = "dev" if split == "validation" else "validation"
    gate = json.loads((root / ART / f"{previous}_validity_gate.json").read_text("utf-8"))
    if not gate["execution_valid"] or gate["protocol_sha256"] != protocol_sha:
        raise ValueError("Previous C3 validity gate not passed")


def ordered_cases(cases, split):
    if split == "dev":
        order = {v: i for i, v in enumerate(SMOKE_IDS)}
        return sorted(
            cases, key=lambda c: (order.get(c["public"]["task_id"], 100), c["public"]["task_id"])
        )
    return sorted(cases, key=lambda c: c["public"]["task_id"])


def call_plan(cases, split, smoke=False):
    cases = ordered_cases(cases, split)
    if smoke:
        cases = [c for c in cases if c["public"]["task_id"] in SMOKE_IDS]
    permutations = list(itertools.permutations(METHODS))
    plan = []
    for i, case in enumerate(cases):
        for method in permutations[i % 6]:
            plan.append((case, method, 0))
    if split == "dev" and not smoke:
        lookup = {c["public"]["task_id"]: c for c in cases}
        for rep in (1, 2):
            for i, task in enumerate(REPEAT_IDS):
                for method in permutations[(i + rep) % 6]:
                    plan.append((lookup[task], method, rep))
    return plan


def summarize_rows(rows):
    groups = {}
    for method in METHODS:
        part = [r for r in rows if r["method"] == method and r["repetition"] == 0]
        if not part:
            continue

        def timing(key, samples=part):
            values = sorted(r["result"]["timings"][key] for r in samples)
            return {
                "median": statistics.median(values),
                "p95": values[math.ceil(0.95 * len(values)) - 1],
                "max": max(values),
            }

        groups[method] = {
            "tasks": len(part),
            "input_tokens": sum(r["result"]["usage"]["input_tokens"] or 0 for r in part),
            "output_tokens": sum(r["result"]["usage"]["output_tokens"] or 0 for r in part),
            "automatic_passes": sum(r["automatic"]["automatic_pass"] for r in part),
            "timings": {
                key: timing(key)
                for key in ("retrieval_seconds", "generation_seconds", "end_to_end_seconds")
            },
        }
    return groups


def prepare_reviews(root: Path, split: str):
    directory = root / RUN / split
    if (directory / "blinded_review.jsonl").exists():
        raise ValueError("Review already prepared")
    _, cases, _ = load_materials(root, split)
    gold = {c["public"]["task_id"]: c for c in cases}
    paths = list(directory.glob("C3-*.json"))
    expected = 120 if split == "dev" else 90
    if len(paths) != expected:
        raise ValueError("Incomplete C3 stage")
    random.SystemRandom().shuffle(paths)
    output = []
    mapping = {}
    for i, path in enumerate(paths, 1):
        row = json.loads(path.read_text("utf-8"))
        case = gold[row["task_id"]]
        rid = f"{split.upper()}-{i:03}"
        output.append(
            {
                "review_id": rid,
                "question": case["public"]["question"],
                "filters": case["public"]["filters"],
                "answer": row["result"]["answer"],
                "error": row["result"]["error"],
                "semantic_checklist": case["criteria"]["semantic_checklist"],
                "expected_status": case["criteria"]["expected_status"],
                "cited_sources": row["result"]["sources"],
            }
        )
        mapping[rid] = {"file": path.name, "sha256": sha(path)}
    (directory / "blinded_review.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in output), "utf-8"
    )
    write_new(directory / "review_mapping.json", mapping)
    return len(output)


def evaluate_review(root: Path, split: str):
    directory = root / RUN / split
    mapping = json.loads((directory / "review_mapping.json").read_text("utf-8"))
    review = json.loads((directory / "semantic_review.json").read_text("utf-8"))
    protocol_sha = sha(root / PROTOCOL_PATH)
    if set(mapping) != set(review):
        raise ValueError("Every response needs explicit review")
    joined = []
    for rid, meta in mapping.items():
        item = review[rid]
        if (
            item["verdict"] not in ("pass", "fail", "needs_review")
            or item["reviewer_kind"] != "assistant"
            or not item["rationale"].strip()
        ):
            raise ValueError("Invalid review")
        path = directory / meta["file"]
        if sha(path) != meta["sha256"]:
            raise ValueError("Reviewed row changed")
        row = json.loads(path.read_text("utf-8"))
        row["semantic_review"] = item
        row["success"] = item["verdict"] == "pass" and row["automatic"]["automatic_pass"]
        joined.append(row)
    technical = [
        r for r in joined if r["result"]["error"] or r["result"]["usage"]["unknown_usage_calls"]
    ]
    unresolved = [r for r in joined if r["semantic_review"]["verdict"] == "needs_review"]
    stats = {}
    for method in METHODS:
        rows = [r for r in joined if r["method"] == method and r["repetition"] == 0]
        stats[method] = {
            "tasks": len(rows),
            "successes": sum(r["success"] for r in rows),
            "family_complete_successes": sum(
                all(r["success"] for r in rows if r["family_id"] == family)
                for family in {r["family_id"] for r in rows}
            ),
            "families": len({r["family_id"] for r in rows}),
            "semantic_passes": sum(r["semantic_review"]["verdict"] == "pass" for r in rows),
            "semantic_failures": sum(r["semantic_review"]["verdict"] == "fail" for r in rows),
            "unresolved": sum(r["semantic_review"]["verdict"] == "needs_review" for r in rows),
        }
    pairs = {}
    for a, b in itertools.combinations(METHODS, 2):
        left = {r["task_id"]: r for r in joined if r["method"] == a and r["repetition"] == 0}
        right = {r["task_id"]: r for r in joined if r["method"] == b and r["repetition"] == 0}
        pairs[a + "_vs_" + b] = {
            "improved": [k for k in left if not left[k]["success"] and right[k]["success"]],
            "regressed": [k for k in left if left[k]["success"] and not right[k]["success"]],
        }
    report = {
        "execution_valid": not technical and not unresolved,
        "protocol_sha256": protocol_sha,
        "split": split,
        "reviewer_kind": "assistant",
        "independent_human_gold": False,
        "review_sha256": sha(directory / "semantic_review.json"),
        "technical_errors": len(technical),
        "unresolved": len(unresolved),
        "statistics": stats,
        "paired": pairs,
        "effect_win_required_to_execute_next_stage": False,
    }
    write_new(root / ART / f"{split}_validity_gate.json", report)
    return report
