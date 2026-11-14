"""请求账本、预算与轨迹的持久化边界；业务订单事务仍由 RetailService 管理。"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path

from aftersalesagent.schemas.contracts import RunLimits, TraceEvent, TurnResult, UserEvent


class EventConflict(ValueError):
    """相同事件ID不能携带不同内容，也不能越过尚未解决的请求。"""


class Journal:
    def __init__(self, path: Path | None = None):
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(
            str(path) if path else ":memory:", check_same_thread=False
        )
        self.connection.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        self.connection.executescript("""
            PRAGMA foreign_keys=ON;
            CREATE TABLE IF NOT EXISTS catalog (
                session_id TEXT PRIMARY KEY, environment TEXT NOT NULL,
                scenario TEXT NOT NULL, token_hash TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS usage (
                session_id TEXT PRIMARY KEY, model_calls INTEGER NOT NULL DEFAULT 0,
                tool_calls INTEGER NOT NULL DEFAULT 0, tokens INTEGER NOT NULL DEFAULT 0,
                unknown_usage_calls INTEGER NOT NULL DEFAULT 0,
                active_seconds REAL NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS requests (
                session_id TEXT NOT NULL, event_id TEXT NOT NULL, fingerprint TEXT NOT NULL,
                event_json TEXT NOT NULL, status TEXT NOT NULL, trace_start INTEGER NOT NULL,
                result_json TEXT, model_calls INTEGER NOT NULL DEFAULT 0,
                tool_calls INTEGER NOT NULL DEFAULT 0, tokens INTEGER NOT NULL DEFAULT 0,
                active_seconds REAL NOT NULL DEFAULT 0,
                PRIMARY KEY (session_id, event_id)
            );
            CREATE TABLE IF NOT EXISTS reservations (
                attempt_id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
                event_id TEXT NOT NULL, seconds REAL NOT NULL, settled INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS trace (
                session_id TEXT NOT NULL, sequence INTEGER NOT NULL,
                event_id TEXT UNIQUE NOT NULL, payload TEXT NOT NULL,
                PRIMARY KEY (session_id, sequence)
            );
        """)
        self.connection.commit()

    @contextmanager
    def transaction(self):
        # 本项目单进程；锁只覆盖短SQLite操作，不跨模型调用或用户等待。
        with self.lock, self.connection:
            yield self.connection

    def close(self) -> None:
        with self.lock:
            self.connection.close()

    def register(self, sid: str) -> None:
        with self.transaction() as c:
            c.execute("INSERT OR IGNORE INTO usage(session_id) VALUES (?)", (sid,))

    def add_session(self, sid: str, environment: str, scenario: str, token: str) -> None:
        with self.transaction() as c:
            c.execute(
                "INSERT INTO catalog VALUES (?, ?, ?, ?)",
                (sid, environment, scenario, hashlib.sha256(token.encode()).hexdigest()),
            )

    def session(self, sid: str) -> dict | None:
        with self.transaction() as c:
            row = c.execute("SELECT * FROM catalog WHERE session_id=?", (sid,)).fetchone()
            return None if row is None else dict(row)

    def claim(self, sid: str, event: UserEvent) -> TurnResult | None:
        payload = event.model_dump_json()
        fingerprint = hashlib.sha256(payload.encode()).hexdigest()
        with self.transaction() as c:
            row = c.execute(
                "SELECT * FROM requests WHERE session_id=? AND event_id=?", (sid, event.event_id)
            ).fetchone()
            if row:
                if row["fingerprint"] != fingerprint:
                    raise EventConflict("event_id already belongs to different request content")
                if row["status"] == "done":
                    return TurnResult.model_validate_json(row["result_json"])
                return None
            other = c.execute(
                "SELECT event_id FROM requests WHERE session_id=? AND status!='done'", (sid,)
            ).fetchone()
            if other:
                raise EventConflict("Retry the unresolved event with its original event_id first")
            trace_start = c.execute(
                "SELECT COALESCE(MAX(sequence),-1) FROM trace WHERE session_id=?", (sid,)
            ).fetchone()[0]
            c.execute(
                "INSERT INTO requests"
                "(session_id,event_id,fingerprint,event_json,status,trace_start)"
                " VALUES (?,?,?,?, 'running',?)",
                (sid, event.event_id, fingerprint, payload, trace_start),
            )
        return None

    def finish(
        self, sid: str, event_id: str, result: TurnResult, *, retryable: bool = False
    ) -> None:
        with self.transaction() as c:
            c.execute(
                "UPDATE requests SET status=?, result_json=? WHERE session_id=? AND event_id=?",
                ("retryable" if retryable else "done", result.model_dump_json(), sid, event_id),
            )

    def request(self, sid: str, event_id: str) -> dict:
        with self.transaction() as c:
            row = c.execute(
                "SELECT * FROM requests WHERE session_id=? AND event_id=?", (sid, event_id)
            ).fetchone()
            if row is None:
                raise ValueError("Unknown event")
            return dict(row)

    def latest(self, sid: str) -> dict | None:
        with self.transaction() as c:
            row = c.execute(
                "SELECT event_json,status,result_json FROM requests"
                " WHERE session_id=? ORDER BY rowid DESC LIMIT 1",
                (sid,),
            ).fetchone()
            if row is None:
                return None
            return {
                "event": json.loads(row["event_json"]),
                "status": row["status"],
                "result": json.loads(row["result_json"]) if row["result_json"] else None,
            }

    def usage(self, sid: str) -> dict:
        with self.transaction() as c:
            row = c.execute("SELECT * FROM usage WHERE session_id=?", (sid,)).fetchone()
            if row is None:
                raise ValueError("Unknown session budget")
            result = dict(row)
            result.pop("session_id")
            result["reported_tokens"] = result.pop("tokens")
            # 未报告usage时总token未知；已知子集仍保留供预算控制。
            result["total_tokens"] = (
                None if result["unknown_usage_calls"] else result["reported_tokens"]
            )
            return result

    def charge(self, sid: str, eid: str, name: str, amount: int = 1) -> None:
        if name not in {"model_calls", "tool_calls", "tokens", "unknown_usage_calls"}:
            raise ValueError("Unknown counter")
        with self.transaction() as c:
            c.execute(f"UPDATE usage SET {name}={name}+? WHERE session_id=?", (amount, sid))
            if name != "unknown_usage_calls":
                c.execute(
                    f"UPDATE requests SET {name}={name}+? WHERE session_id=? AND event_id=?",
                    (amount, sid, eid),
                )

    def reserve(self, sid: str, eid: str, limits: RunLimits) -> tuple[str, float]:
        with self.transaction() as c:
            total = c.execute(
                "SELECT active_seconds FROM usage WHERE session_id=?", (sid,)
            ).fetchone()[0]
            turn = c.execute(
                "SELECT active_seconds FROM requests WHERE session_id=? AND event_id=?", (sid, eid)
            ).fetchone()[0]
            seconds = max(
                0.0,
                min(limits.max_active_seconds - total, limits.max_turn_active_seconds - turn),
            )
            aid = uuid.uuid4().hex
            # 先预留再执行。进程突然死亡时保守保留预算，不把未知执行时间当作零。
            c.execute("INSERT INTO reservations VALUES (?,?,?,?,0)", (aid, sid, eid, seconds))
            c.execute(
                "UPDATE usage SET active_seconds=active_seconds+? WHERE session_id=?",
                (seconds, sid),
            )
            c.execute(
                "UPDATE requests SET active_seconds=active_seconds+?"
                " WHERE session_id=? AND event_id=?",
                (seconds, sid, eid),
            )
            return aid, seconds

    def settle(self, attempt: str, actual: float) -> None:
        with self.transaction() as c:
            row = c.execute("SELECT * FROM reservations WHERE attempt_id=?", (attempt,)).fetchone()
            if row is None or row["settled"]:
                return
            # 同步事务不可抢占；实际工作超时也要如实计入。
            delta = max(0.0, actual) - row["seconds"]
            c.execute(
                "UPDATE usage SET active_seconds=active_seconds+? WHERE session_id=?",
                (delta, row["session_id"]),
            )
            c.execute(
                "UPDATE requests SET active_seconds=active_seconds+?"
                " WHERE session_id=? AND event_id=?",
                (delta, row["session_id"], row["event_id"]),
            )
            c.execute("UPDATE reservations SET settled=1 WHERE attempt_id=?", (attempt,))

    def append(self, sid: str, kind: str, payload: dict, created_at) -> TraceEvent:
        with self.transaction() as c:
            sequence = c.execute(
                "SELECT COALESCE(MAX(sequence),-1)+1 FROM trace WHERE session_id=?", (sid,)
            ).fetchone()[0]
            event = TraceEvent.model_validate(
                {
                    "event_id": uuid.uuid4().hex,
                    "session_id": sid,
                    "sequence": sequence,
                    "kind": kind,
                    "payload": redact(payload),
                    "created_at": created_at,
                }
            )
            c.execute(
                "INSERT INTO trace VALUES (?,?,?,?)",
                (sid, sequence, event.event_id, event.model_dump_json()),
            )
            return event

    def events(self, sid: str, after: int = -1) -> list[TraceEvent]:
        with self.transaction() as c:
            return [
                TraceEvent.model_validate_json(row[0])
                for row in c.execute(
                    "SELECT payload FROM trace WHERE session_id=? AND sequence>? ORDER BY sequence",
                    (sid, after),
                )
            ]

    def export(self, sid: str, directory: Path) -> Path:
        with self.lock:
            return self._export(sid, directory)

    def _export(self, sid: str, directory: Path) -> Path:
        # SQLite是权威轨迹，JSONL是可重建的投影。中途写坏/丢失文件不重跑模型。
        if len(sid) != 32 or any(ch not in "0123456789abcdef" for ch in sid):
            raise ValueError("Invalid session identifier")
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (sid + ".jsonl")
        temporary = directory / (sid + ".jsonl.tmp")
        temporary.write_text(
            "".join(event.model_dump_json() + "\n" for event in self.events(sid)), encoding="utf-8"
        )
        temporary.replace(path)
        return path


def redact(value):
    if isinstance(value, dict):
        return {
            key: "[REDACTED]"
            if key.lower().replace("-", "_")
            in {
                "authorization",
                "api_key",
                "access_token",
                "token",
                "cookie",
                "set_cookie",
                "deepseek_api_key",
            }
            else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value
