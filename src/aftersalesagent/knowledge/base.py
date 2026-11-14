"""可替换知识检索接口；E1保留来源、证据ID与版本。"""

from typing import Protocol

from aftersalesagent.knowledge.corpus import KnowledgeChunk
from aftersalesagent.schemas.contracts import Contract


class EvidenceChunk(Contract):
    document_id: str
    section: str
    text: str
    version: str
    source: str
    score: float
    chunk_id: str | None = None
    scope: str | None = None
    origin: str | None = None
    retrieval_method: str | None = None
    score_direction: str | None = None
    index_fingerprint: str | None = None


class KnowledgeRetriever(Protocol):
    async def search(
        self, query: str, filters: dict[str, str], top_k: int
    ) -> list[EvidenceChunk]: ...


class PolicyEvidenceProvider(Protocol):
    tokenizer: str

    def select(self, filters: dict[str, str]) -> list[KnowledgeChunk]: ...

    async def provide(
        self, query: str, filters: dict[str, str], top_k: int, *, full: bool = False
    ) -> list[KnowledgeChunk]: ...

    def close(self) -> None: ...
