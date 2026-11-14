"""SQLite FTS5/BM25；三种实际索引共用同一语料与元数据过滤。"""

import re
import sqlite3

from aftersalesagent.knowledge.base import EvidenceChunk
from aftersalesagent.knowledge.corpus import KnowledgeChunk, eligible


def lexical_tokens(text: str, mode: str) -> list[str]:
    parts = re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]+", text.lower())
    tokens = []
    for part in parts:
        chinese = "\u4e00" <= part[0] <= "\u9fff"
        if not chinese or mode == "unicode61":
            if mode != "trigram" or len(part) >= 3:
                tokens.append(part)
        elif mode == "trigram":
            tokens += [part[i : i + 3] for i in range(len(part) - 2)]
        else:
            # 显式字符1/2-gram预分词；是词法检索，不是中文语义分词或向量检索。
            tokens += list(part)
            tokens += [part[i : i + 2] for i in range(len(part) - 1)]
    return list(dict.fromkeys(tokens))


class SQLiteKnowledgeIndex:
    def __init__(self, chunks: list[KnowledgeChunk], tokenizer="char", path=":memory:"):
        if tokenizer not in {"unicode61", "trigram", "char"}:
            raise ValueError("Unknown tokenizer")
        self.chunks = chunks
        self.by_id = {c.chunk_id: c for c in chunks}
        self.tokenizer = tokenizer
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.execute("DROP TABLE IF EXISTS evidence_fts")
        self.connection.execute(
            "CREATE VIRTUAL TABLE evidence_fts USING fts5(chunk_id UNINDEXED, body, tokenize='"
            + ("trigram" if tokenizer == "trigram" else "unicode61")
            + "')"
        )
        for c in chunks:
            text = c.section + " " + c.text
            if tokenizer == "char":
                text = " ".join(lexical_tokens(text, "char"))
            self.connection.execute("INSERT INTO evidence_fts VALUES (?,?)", (c.chunk_id, text))
        self.connection.commit()

    def close(self):
        self.connection.close()

    def select(self, filters: dict[str, str]) -> list[KnowledgeChunk]:
        return eligible(self.chunks, filters)

    def ranked(
        self, query: str, filters: dict[str, str], top_k: int
    ) -> list[tuple[KnowledgeChunk, float]]:
        if not 1 <= top_k <= 30:
            raise ValueError("top_k must be 1..30")
        allowed = {c.chunk_id for c in self.select(filters)}
        tokens = lexical_tokens(query, self.tokenizer)
        if not tokens or not allowed:
            return []
        match = " OR ".join('"' + t.replace('"', '""') + '"' for t in tokens)
        # 过滤在截断top-k之前执行；不让其他商家/版本挤掉正确范围的结果。
        ids = sorted(allowed)
        rows = self.connection.execute(
            "SELECT chunk_id,bm25(evidence_fts) AS score FROM evidence_fts "
            "WHERE evidence_fts MATCH ? AND chunk_id IN ("
            + ",".join("?" for _ in ids)
            + ") ORDER BY score,chunk_id LIMIT ?",
            (match, *ids, top_k),
        ).fetchall()
        return [(self.by_id[row[0]], float(row[1])) for row in rows]

    def retrieve(self, query: str, filters: dict[str, str], top_k: int) -> list[KnowledgeChunk]:
        return [chunk for chunk, _ in self.ranked(query, filters, top_k)]

    async def provide(self, query, filters, top_k, *, full=False):
        return self.select(filters) if full else self.retrieve(query, filters, top_k)

    async def search(self, query: str, filters: dict[str, str], top_k: int) -> list[EvidenceChunk]:
        return [
            EvidenceChunk(
                document_id=c.document_id,
                chunk_id=c.chunk_id,
                section=c.section,
                text=c.text,
                version=c.version,
                source=c.source,
                scope=c.scope,
                origin=c.origin,
                score=score,
            )
            for c, score in self.ranked(query, filters, top_k)
        ]
