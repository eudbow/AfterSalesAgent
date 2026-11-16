"""有版本的证据卡片与官方原文定位；资料快照不冒充政策生效日期。"""

import hashlib
import json
from datetime import date
from pathlib import Path

from aftersalesagent.schemas.contracts import Contract


class KnowledgeChunk(Contract):
    chunk_id: str
    document_id: str
    section: str
    text: str
    version: str
    source: str
    scope: str
    origin: str
    category: str = "all"
    effective_from: str | None = None
    effective_until: str | None = None
    source_text_path: str
    source_start: int
    source_end: int
    source_excerpt_sha256: str


def dump(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def load_corpus(root: Path, version: str = "e1_v1") -> tuple[list[KnowledgeChunk], dict]:
    if version not in {"e1_v1", "e4_v1", "retail_sim_v1"}:
        raise ValueError("Unknown corpus version")
    manifest = json.loads((root / f"data/knowledge/{version}/manifest.json").read_text("utf-8"))
    for rel, sha in manifest["files"].items():
        p = (root / rel).resolve()
        if (
            not p.is_relative_to(root.resolve())
            or hashlib.sha256(p.read_bytes()).hexdigest() != sha
        ):
            raise ValueError("Knowledge snapshot hash mismatch")
    chunks = [
        KnowledgeChunk.model_validate_json(line)
        for line in (root / f"data/knowledge/{version}/chunks.jsonl").read_bytes().splitlines()
    ]
    if len({c.chunk_id for c in chunks}) != len(chunks):
        raise ValueError("Duplicate evidence IDs")
    for c in chunks:
        p = (root / c.source_text_path).resolve()
        if not p.is_relative_to(root.resolve()):
            raise ValueError("Knowledge source escapes corpus")
        excerpt = p.read_text("utf-8")[c.source_start : c.source_end]
        if hashlib.sha256(excerpt.encode()).hexdigest() != c.source_excerpt_sha256:
            raise ValueError("Evidence source span mismatch")
        if (
            version == "e1_v1"
            and c.origin == "official"
            and (c.effective_from or c.effective_until)
        ):
            raise ValueError("Official effective dates must remain unknown in E1 v1")
    return chunks, manifest


def eligible(chunks: list[KnowledgeChunk], filters: dict[str, str]) -> list[KnowledgeChunk]:
    if set(filters) - {"scope", "category", "as_of"}:
        raise ValueError("Unsupported knowledge filter")
    scope = filters.get("scope", "ikea_cn")
    if scope not in {"ikea_cn", "demo_store", "demo_conflict", "retail_sim"}:
        raise ValueError("Unknown knowledge scope")
    as_of = date.fromisoformat(filters["as_of"]) if filters.get("as_of") else None
    selected = []
    for c in chunks:
        if c.scope != scope:
            continue
        if filters.get("category") and c.category not in {"all", filters["category"]}:
            continue
        if as_of:
            # 日期明确时，未知生效日期的官方快照不能用来推断历史适用性。
            if c.effective_from is None:
                continue
            if as_of < date.fromisoformat(c.effective_from):
                continue
            if c.effective_until and as_of > date.fromisoformat(c.effective_until):
                continue
        selected.append(c)
    return sorted(selected, key=lambda c: c.chunk_id)
