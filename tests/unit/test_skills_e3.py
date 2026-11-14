"""版本库是受信本机资产；哈希校验、不可覆盖、原子指针及输入边界。"""

import asyncio
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from pydantic import ValidationError

from aftersalesagent.config import Settings
from aftersalesagent.schemas.contracts import RunContext
from aftersalesagent.skills.base import NullSkillProvider, text_hash
from aftersalesagent.skills.registry import SkillRegistry

ROOT = Path(__file__).resolve().parents[2]
SKILL_ID = "after-sales-workflow"


def make_bundle(
    tmp_path, *, version="1.0.1", parent: str | None = "1.0.0", text_suffix="\n测试专用版本。\n"
):
    base = ROOT / "skills" / SKILL_ID / "1.0.0"
    payload = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    text = (base / "SKILL.md").read_bytes().decode("utf-8") + text_suffix
    payload.update(
        version=version,
        parent_version=parent,
        content_sha256=text_hash(text),
        change_notes="仅供版本机制测试，不是候选收益或生产发布。",
    )
    folder = tmp_path / "bundles" / SKILL_ID / version
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    (folder / "SKILL.md").write_bytes(text.encode("utf-8"))
    return folder.parents[1]


def registry(tmp_path):
    result = SkillRegistry(tmp_path / "registry.sqlite")
    result.install(ROOT / "skills", SKILL_ID, "1.0.0")
    return result


def test_install_list_round_trip_and_source_file_independence(tmp_path):
    assets = make_bundle(tmp_path, version="2.0.0", parent=None)
    reg = SkillRegistry(tmp_path / "registry.sqlite")
    original = reg.install(assets, SKILL_ID, "2.0.0")
    assert reg.install(assets, SKILL_ID, "2.0.0") == original
    path = assets / SKILL_ID / "2.0.0" / "SKILL.md"
    path.write_text("本地原件被修改，不应影响已安装版本。", encoding="utf-8")
    reopened = SkillRegistry(reg.path)
    assert asyncio.run(reopened.load(SKILL_ID, "2.0.0")) == original
    assert asyncio.run(reopened.list_metadata()) == [original.metadata()]
    with pytest.raises(ValueError, match="hash"):
        reg.install(assets, SKILL_ID, "2.0.0")


def test_same_version_cannot_be_replaced_even_with_valid_new_hash(tmp_path):
    assets = make_bundle(tmp_path, version="1.0.0", parent=None)
    reg = registry(tmp_path)
    with pytest.raises(ValueError, match="immutable"):
        reg.install(assets, SKILL_ID, "1.0.0")


@pytest.mark.parametrize("field", ["text", "identity", "unknown_metadata"])
def test_bundle_hash_identity_and_metadata_validation(tmp_path, field):
    assets = make_bundle(tmp_path, parent=None)
    folder = assets / SKILL_ID / "1.0.1"
    if field == "text":
        with (folder / "SKILL.md").open("ab") as stream:
            stream.write(b"modified")
    else:
        payload = json.loads((folder / "manifest.json").read_text())
        payload["version" if field == "identity" else "allow_skip_confirmation"] = (
            "7.0.0" if field == "identity" else True
        )
        (folder / "manifest.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        SkillRegistry(tmp_path / "registry.sqlite").install(assets, SKILL_ID, "1.0.1")


@pytest.mark.parametrize(
    ("skill_id", "version"),
    [("../escape", "1.0.0"), (SKILL_ID, "../1.0.0"), (SKILL_ID, "none"), ("UPPER", "1.0.0")],
)
def test_invalid_refs_rejected_before_reading_files(tmp_path, skill_id, version):
    reg = SkillRegistry(tmp_path / "registry.sqlite")
    with pytest.raises(ValueError, match="Invalid"):
        reg.install(tmp_path / "missing-root", skill_id, version)
    with pytest.raises(ValueError, match="Invalid"):
        asyncio.run(reg.load(skill_id, version))


def test_missing_parent_and_unknown_version_fail(tmp_path):
    assets = make_bundle(tmp_path)
    reg = SkillRegistry(tmp_path / "registry.sqlite")
    with pytest.raises(ValueError, match="Unknown"):
        reg.install(assets, SKILL_ID, "1.0.1")
    with pytest.raises(ValueError, match="Unknown"):
        asyncio.run(reg.load(SKILL_ID, "9.0.0"))
    with pytest.raises(ValueError, match="No manually activated"):
        asyncio.run(reg.load(SKILL_ID, "released"))


def test_atomic_default_switch_rollback_and_audit(tmp_path):
    reg = registry(tmp_path)
    reg.install(make_bundle(tmp_path), SKILL_ID, "1.0.1")
    reg.activate(SKILL_ID, "1.0.0", expected_current="none", reason="人工固定基线")
    with pytest.raises(ValueError, match="changed"):
        reg.activate(SKILL_ID, "1.0.1", expected_current="none", reason="过期写入")
    reg.activate(SKILL_ID, "1.0.1", expected_current="1.0.0", reason="仅测试切换")
    reg.activate(SKILL_ID, "1.0.0", expected_current="1.0.1", reason="人工回退")
    assert asyncio.run(reg.load(SKILL_ID, "released")).version == "1.0.0"
    assert [x["version"] for x in reg.history()] == ["1.0.0", "1.0.1", "1.0.0"]
    assert reg.history()[-1]["previous_version"] == "1.0.1"
    with pytest.raises(ValueError, match="already active"):
        reg.activate(SKILL_ID, "1.0.0", expected_current="1.0.0", reason="重复")
    assert len(reg.history()) == 3


def test_concurrent_pointer_updates_only_one_can_succeed(tmp_path):
    reg = registry(tmp_path)
    for version in ("1.0.1", "1.0.2"):
        reg.install(make_bundle(tmp_path, version=version), SKILL_ID, version)
    reg.activate(SKILL_ID, "1.0.0", expected_current="none", reason="start")

    def switch(version):
        try:
            reg.activate(SKILL_ID, version, expected_current="1.0.0", reason="race-test")
            return True
        except ValueError:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(switch, ["1.0.1", "1.0.2"])) == [False, True]
    assert len(reg.history()) == 2


def test_stored_payload_corruption_rejected(tmp_path):
    reg = registry(tmp_path)
    with sqlite3.connect(reg.path) as db:
        db.execute("UPDATE skill_versions SET payload=payload || ' '")
    with pytest.raises(ValueError, match="payload hash"):
        asyncio.run(reg.load(SKILL_ID, "1.0.0"))


@pytest.mark.parametrize(
    "fields",
    [
        {"skill_version": "1.0.0"},
        {"skill_id": SKILL_ID},
        {"skill_id": "../bad", "skill_version": "1.0.0"},
        {"skill_id": SKILL_ID, "skill_version": "none"},
        {"skill_sha256": "0" * 64},
    ],
)
def test_incoherent_configuration_is_rejected(fields):
    with pytest.raises(ValidationError):
        RunContext(environment_id="test", **fields)
    if "skill_sha256" not in fields:
        with pytest.raises(ValidationError):
            Settings(**fields)


def test_null_provider_and_explicit_environment_selection():
    assert asyncio.run(NullSkillProvider().list_metadata()) == []
    with pytest.raises(ValueError, match="No Skill"):
        asyncio.run(NullSkillProvider().load(SKILL_ID, "1.0.0"))
    config = Settings.from_environment(
        {"AFTERSALES_SKILL_ID": SKILL_ID, "AFTERSALES_SKILL_VERSION": "released"}
    )
    assert config.skill_id == SKILL_ID and config.skill_version == "released"
    assert Settings.from_environment({}).skill_version == "none"
