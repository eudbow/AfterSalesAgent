"""Durable global E4 paid-call reservations; unknown consumption prevents automatic continuation."""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path


class GenerationBudget:
    def __init__(self, path: Path, limit: int = 300):
        if not 1 <= limit <= 300:
            raise ValueError("E4 generation limit must be 1..300")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.limit = limit
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS calls (call_key TEXT PRIMARY KEY, "
            "run_id TEXT NOT NULL, state TEXT NOT NULL, metadata TEXT NOT NULL)"
        )
        self.db.commit()

    def reserve(self, call_key: str, run_id: str):
        try:
            self.db.execute("BEGIN IMMEDIATE")
            if self.db.execute("SELECT 1 FROM calls WHERE state != 'known' LIMIT 1").fetchone():
                raise ValueError("Unresolved/unknown E4 usage exists; no automatic retry")
            if self.db.execute("SELECT count(*) FROM calls").fetchone()[0] >= self.limit:
                raise ValueError("E4 generation request budget exhausted")
            self.db.execute(
                "INSERT INTO calls VALUES (?,?,'pending',?)",
                (call_key, run_id, json.dumps({"reserved_at": datetime.now(UTC).isoformat()})),
            )
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

    def finish(self, call_key: str, *, input_tokens, output_tokens, error=None, model=None):
        known = type(input_tokens) is int and type(output_tokens) is int
        metadata = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "error": error,
            "model": model,
            "finished_at": datetime.now(UTC).isoformat(),
        }
        cursor = self.db.execute(
            "UPDATE calls SET state=?, metadata=? WHERE call_key=? AND state='pending'",
            ("known" if known else "unknown", json.dumps(metadata), call_key),
        )
        if cursor.rowcount != 1:
            raise ValueError("Paid-call reservation missing or already settled")
        self.db.commit()

    def report(self):
        return {
            "limit": self.limit,
            "reserved_calls": self.db.execute("SELECT count(*) FROM calls").fetchone()[0],
            "states": dict(self.db.execute("SELECT state,count(*) FROM calls GROUP BY state")),
        }

    def close(self):
        self.db.close()
