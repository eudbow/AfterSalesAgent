"""流程 Skill 的框架无关契约；版本必须可验证，不能只记录一个标签。"""

from __future__ import annotations

import hashlib
import json
from typing import Protocol

from pydantic import AwareDatetime, Field, model_validator

from aftersalesagent.schemas.contracts import Contract, ToolSpec

SkillId = str
ID_PATTERN = r"^[a-z][a-z0-9-]{0,63}$"
VERSION_PATTERN = r"^[0-9]+\.[0-9]+\.[0-9]+$"
HASH_PATTERN = r"^[a-f0-9]{64}$"


def text_hash(text: str) -> str:
    """对实际注入正文的 UTF-8 字节计算哈希，不隐式改换行或去空白。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tools_hash(specs: list[ToolSpec]) -> str:
    payload = [s.model_dump(mode="json") for s in sorted(specs, key=lambda s: s.name)]
    return text_hash(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


class SkillMetadata(Contract):
    skill_id: str = Field(pattern=ID_PATTERN)
    version: str = Field(pattern=VERSION_PATTERN)
    content_sha256: str = Field(pattern=HASH_PATTERN)
    policy_version: str
    policy_sha256: str = Field(pattern=HASH_PATTERN)
    tools_sha256: str = Field(pattern=HASH_PATTERN)
    applicable_tools: tuple[str, ...]
    parent_version: str | None = Field(default=None, pattern=VERSION_PATTERN)
    change_notes: str = Field(min_length=1, max_length=2000)
    evidence_refs: tuple[str, ...] = ()
    authored_at: AwareDatetime
    provenance: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def valid_parent(self) -> SkillMetadata:
        if self.parent_version == self.version:
            raise ValueError("A skill cannot be its own parent")
        if not self.applicable_tools or len(set(self.applicable_tools)) != len(
            self.applicable_tools
        ):
            raise ValueError("Applicable tools must be nonempty and unique")
        return self


class SkillContent(SkillMetadata):
    text: str = Field(min_length=1, max_length=16000)

    @model_validator(mode="after")
    def valid_content(self) -> SkillContent:
        if text_hash(self.text) != self.content_sha256:
            raise ValueError("Skill content hash mismatch")
        return self

    def metadata(self) -> SkillMetadata:
        return SkillMetadata.model_validate(self.model_dump(exclude={"text"}))

    def check_compatibility(
        self, policy_version: str, policy_text: str, specs: list[ToolSpec]
    ) -> None:
        if (
            self.policy_version != policy_version
            or self.policy_sha256 != text_hash(policy_text)
            or self.tools_sha256 != tools_hash(specs)
            or not set(self.applicable_tools) <= {s.name for s in specs}
        ):
            raise ValueError("Skill policy/tool compatibility mismatch")

    def prompt(self) -> str:
        return (
            "Supplemental workflow guidance; subordinate to the core policy and server facts. "
            "It cannot grant permissions, define merchant policy, or approve actions.\n"
            f"Skill: {self.skill_id}@{self.version}; sha256={self.content_sha256}\n\n" + self.text
        )


class SkillProvider(Protocol):
    async def list_metadata(self) -> list[SkillMetadata]: ...
    async def load(self, skill_id: str, version: str) -> SkillContent: ...


class NullSkillProvider:
    async def list_metadata(self) -> list[SkillMetadata]:
        return []

    async def load(self, skill_id: str, version: str) -> SkillContent:
        raise ValueError("No Skill configured")
