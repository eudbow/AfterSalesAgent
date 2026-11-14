"""人工导入、不可覆盖的版本库与原子默认指针；不包含候选生成或自动发布。"""

from __future__ import annotations

import asyncio
import re
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from aftersalesagent.skills.base import (
    ID_PATTERN,
    VERSION_PATTERN,
    SkillContent,
    SkillMetadata,
    text_hash,
)


def validate_ref(skill_id: str, version: str, *, released: bool = False) -> None:
    if not re.fullmatch(ID_PATTERN, skill_id) or not (
        re.fullmatch(VERSION_PATTERN, version) or (released and version == "released")
    ):
        raise ValueError("Invalid Skill ID/version")


class SkillRegistry:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._transaction() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS skill_versions (
                    skill_id TEXT NOT NULL, version TEXT NOT NULL,
                    payload TEXT NOT NULL, payload_sha256 TEXT NOT NULL,
                    PRIMARY KEY(skill_id, version)
                );
                CREATE TABLE IF NOT EXISTS skill_defaults (
                    skill_id TEXT PRIMARY KEY, version TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS skill_decisions (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    skill_id TEXT NOT NULL, previous_version TEXT NOT NULL,
                    version TEXT NOT NULL, reason TEXT NOT NULL, created_at TEXT NOT NULL
                );
            """)

    @contextmanager
    def _transaction(self):
        # 每次只持有短事务，不跨模型调用；BEGIN IMMEDIATE 保护指针的比较再交换。
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def install(self, bundle_root: Path, skill_id: str, version: str) -> SkillContent:
        validate_ref(skill_id, version)
        root = bundle_root.resolve(strict=True)
        directory = (root / skill_id / version).resolve(strict=True)
        if not directory.is_relative_to(root):
            raise ValueError("Skill bundle escapes its root")
        paths = [(directory / name).resolve(strict=True) for name in ("manifest.json", "SKILL.md")]
        if any(not p.is_relative_to(directory) for p in paths):
            raise ValueError("Skill file escapes its version directory")
        metadata = SkillMetadata.model_validate_json(paths[0].read_bytes())
        # read_bytes 保留正文原换行；哈希针对实际字节。仅接受严格 UTF-8。
        content = SkillContent(**metadata.model_dump(), text=paths[1].read_bytes().decode("utf-8"))
        if (content.skill_id, content.version) != (skill_id, version):
            raise ValueError("Bundle identity mismatch")
        payload = content.model_dump_json()
        digest = text_hash(payload)
        with self._transaction() as db:
            existing = db.execute(
                "SELECT payload_sha256 FROM skill_versions WHERE skill_id=? AND version=?",
                (skill_id, version),
            ).fetchone()
            if existing:
                if existing["payload_sha256"] != digest:
                    raise ValueError("An installed Skill version is immutable; use a new version")
                return content
            if content.parent_version:
                self._read(db, skill_id, content.parent_version)
            db.execute(
                "INSERT INTO skill_versions VALUES (?,?,?,?)", (skill_id, version, payload, digest)
            )
        return content

    @staticmethod
    def _read(db: sqlite3.Connection, skill_id: str, version: str) -> SkillContent:
        row = db.execute(
            "SELECT payload, payload_sha256 FROM skill_versions WHERE skill_id=? AND version=?",
            (skill_id, version),
        ).fetchone()
        if row is None:
            raise ValueError("Unknown Skill version")
        if text_hash(row["payload"]) != row["payload_sha256"]:
            raise ValueError("Stored Skill payload hash mismatch")
        return SkillContent.model_validate_json(row["payload"])

    async def load(self, skill_id: str, version: str) -> SkillContent:
        validate_ref(skill_id, version, released=True)

        def read():
            with self._transaction() as db:
                selected = version
                if selected == "released":
                    row = db.execute(
                        "SELECT version FROM skill_defaults WHERE skill_id=?", (skill_id,)
                    ).fetchone()
                    if row is None:
                        raise ValueError("No manually activated Skill version")
                    selected = row["version"]
                return self._read(db, skill_id, selected)

        return await asyncio.to_thread(read)

    async def list_metadata(self) -> list[SkillMetadata]:
        def read():
            with self._transaction() as db:
                rows = db.execute(
                    "SELECT skill_id,version FROM skill_versions ORDER BY skill_id,version"
                ).fetchall()
                return [self._read(db, r["skill_id"], r["version"]).metadata() for r in rows]

        return await asyncio.to_thread(read)

    def activate(self, skill_id: str, version: str, *, expected_current: str, reason: str) -> dict:
        """人工切换/回退共用同一原子操作；不代表通过了 E3 自动优化发布门槛。"""
        validate_ref(skill_id, version)
        if not reason.strip() or len(reason) > 2000:
            raise ValueError("An activation reason is required")
        with self._transaction() as db:
            self._read(db, skill_id, version)
            row = db.execute(
                "SELECT version FROM skill_defaults WHERE skill_id=?", (skill_id,)
            ).fetchone()
            previous = row["version"] if row else "none"
            if previous != expected_current:
                raise ValueError("Default Skill changed; inspect it before retrying")
            if previous == version:
                raise ValueError("Skill version is already active")
            db.execute(
                "INSERT INTO skill_defaults VALUES (?,?) "
                "ON CONFLICT(skill_id) DO UPDATE SET version=excluded.version",
                (skill_id, version),
            )
            decision = {
                "skill_id": skill_id,
                "previous_version": previous,
                "version": version,
                "reason": reason,
                "created_at": datetime.now(UTC).isoformat(),
            }
            db.execute(
                "INSERT INTO skill_decisions "
                "(skill_id,previous_version,version,reason,created_at) VALUES (?,?,?,?,?)",
                tuple(decision.values()),
            )
            return decision

    def history(self) -> list[dict]:
        with self._transaction() as db:
            return [dict(r) for r in db.execute("SELECT * FROM skill_decisions ORDER BY sequence")]
