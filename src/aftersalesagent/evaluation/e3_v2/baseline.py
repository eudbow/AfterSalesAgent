"""v2真实开发基线：48个预定episode；验证/测试只封存，不允许本入口真实执行。"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from aftersalesagent.business.database import RetailStore
from aftersalesagent.business.service import RetailService
from aftersalesagent.evaluation.e3_v2.contracts import SimulatorOptions
from aftersalesagent.evaluation.e3_v2.holdout import load as load_holdout
from aftersalesagent.evaluation.e3_v2.replay import ProtocolReplay
from aftersalesagent.evaluation.e3_v2.review import export_review, save
from aftersalesagent.evaluation.e3_v2.runner import run_case
from aftersalesagent.evaluation.e3_v2.scoring import aggregate
from aftersalesagent.evaluation.e3_v2.suite import load as load_dev
from aftersalesagent.evaluation.e3_v2.suite import load_protocol, sha
from aftersalesagent.evaluation.e3_v2.user import FormalUser
from aftersalesagent.local_config import options_from_settings
from aftersalesagent.models.deepseek import DeepSeekBackend
from aftersalesagent.policies.application import load_application_policy
from aftersalesagent.schemas.contracts import RunContext, RunLimits
from aftersalesagent.skills.base import tools_hash
from aftersalesagent.skills.registry import SkillRegistry


def freeze_config(root: Path, settings):
    _, dev = load_dev(root)
    holdout = json.loads((root / "data/manifests/e3_v2_holdout_v1.json").read_text("utf-8"))
    template = root / ".local/databases/retail_template.sqlite"
    store = RetailStore(template)
    try:
        schema_hash = tools_hash(RetailService(store).specs())
    finally:
        store.close()
    paths = [
        *root.joinpath("src").rglob("*.py"),
        root / "scripts/run_e3_v2_baseline.py",
        root / "pyproject.toml",
        root / "uv.lock",
        root / "configs/m3_live_limits.json",
        root / "configs/evaluation/e3_protocol_v2.json",
        *root.joinpath("skills/after-sales-workflow/1.0.0").glob("*"),
    ]
    return {
        "experiment_id": "e3-v2-real-dev-baseline-v1",
        "planned_on": "2026-10-27",
        "selected_split": "dev",
        "episodes": 48,
        "repetitions": 2,
        "maximum_model_requests": 576,
        "transport": "mcp",
        "options": options_from_settings(settings).model_dump(mode="json"),
        "limits": json.loads((root / "configs/m3_live_limits.json").read_text("utf-8")),
        "protocol": load_protocol(root),
        "protocol_sha256": sha(root / "configs/evaluation/e3_protocol_v2.json"),
        "dev_manifest_sha256": sha(root / "data/manifests/e3_v2_dev_suite.json"),
        "dev_files": dev["files"],
        "sealed_holdout_manifest_sha256": sha(root / "data/manifests/e3_v2_holdout_v1.json"),
        "sealed_holdout_files": holdout["files"],
        "policy_sha256": hashlib.sha256(load_application_policy(root).encode()).hexdigest(),
        "tools_sha256": schema_hash,
        "database_template_sha256": sha(template),
        "groups": {
            "S0": {"skill_id": None, "skill_version": "none"},
            "S1": {"skill_id": "after-sales-workflow", "skill_version": "1.0.0"},
        },
        "order": "alternate group first by task index plus repetition",
        "smoke": {"task_id": "e3v2-dev-01", "repetition": 1, "included_in_total": True},
        "smoke_gate": (
            "both read expected authenticated order, complete provider usage, no provider errors"
        ),
        "sources": {p.relative_to(root).as_posix(): sha(p) for p in sorted(paths) if p.is_file()},
        "holdout_policy": (
            "No live validation/test execution or result inspection in this baseline run"
        ),
        "language_review": (
            "blind sheet; explicit reviewer; assistant reviews are not independent human truth"
        ),
    }


def rows_from(directory: Path):
    episodes = []
    for path in sorted(directory.glob("*/result.json")):
        parts = path.parent.name.rsplit("_", 2)
        if len(parts) != 3 or not parts[1].startswith("r"):
            continue
        episodes.append(
            {
                "episode_id": path.parent.name,
                "repetition": int(parts[1][1:]),
                "result": json.loads(path.read_text("utf-8")),
            }
        )
    return episodes


def smoke_gate(directory: Path):
    rows = [
        e
        for e in rows_from(directory)
        if e["result"]["task_id"] == "e3v2-dev-01" and e["repetition"] == 1
    ]
    ready = len(rows) == 2
    ids = []
    for entry in rows:
        row = entry["result"]
        events = [
            json.loads(s)
            for s in (directory / entry["episode_id"] / "trace.jsonl")
            .read_text("utf-8")
            .splitlines()
        ]
        responses = [event["payload"] for event in events if event["kind"] == "model_response"]
        ids += [response.get("provider_metadata", {}).get("id") for response in responses]
        ready &= (
            row["scoring"]["business"]["checks"]["read_expected_order"]
            and row["scoring"]["business"]["checks"]["located_expected_user"]
            and row["usage"]["unknown_usage_calls"] == 0
            and len(responses) == row["usage"]["model_calls"] > 0
            and not any(event["kind"] == "model_error" for event in events)
            and row["terminal_status"] != "failed"
            and all(
                response["input_tokens"] is not None and response["output_tokens"] is not None
                for response in responses
            )
        )
    ready &= bool(ids) and all(ids) and len(ids) == len(set(ids))
    return {
        "ready": bool(ready),
        "episodes": len(rows),
        "provider_responses": len(ids),
        "task_status": [e["result"]["scoring"]["task_status"] for e in rows],
        "note": "Connectivity/usage gate; semantic task success is not a selection filter.",
    }


async def live(root: Path, *, directory: Path, phase, settings, secret, config):
    if phase not in {"smoke", "remaining"}:
        raise ValueError("Live mode supports only staged development smoke/remaining")
    cases, _ = load_dev(root)  # 不加载留出脚本/答案；只核对它们的哈希。
    if not directory.resolve().is_relative_to((root / ".local").resolve()):
        raise ValueError("Live output must stay in project .local")
    directory.mkdir(parents=True, exist_ok=True)
    saved = directory / "config.json"
    if saved.exists() and json.loads(saved.read_text("utf-8")) != config:
        raise ValueError("Run config mismatch")
    if not saved.exists():
        save(saved, config)
    metadata = directory / "run_metadata.json"
    if not metadata.exists():
        save(
            metadata,
            {
                "started_at_utc": datetime.now(UTC).isoformat(),
                "planned_on": config["planned_on"],
                "selected_split": "dev",
            },
        )
    if phase == "remaining":
        gate_path = directory / "smoke_gate.json"
        if not gate_path.exists() or not smoke_gate(directory)["ready"]:
            raise ValueError("Complete the frozen two-episode smoke first")
    protocol = config["protocol"]
    options = SimulatorOptions.model_validate(
        {
            key: protocol["simulator"][key]
            for key in ("per_slot_per_stage", "per_stage", "total_optional_replies")
        }
    )
    skill = SkillRegistry(directory / "skills.sqlite")
    skill.install(root / "skills", "after-sales-workflow", "1.0.0")
    limits = RunLimits.model_validate(config["limits"])
    for index, case in enumerate(cases, 1):
        for repetition in (1, 2):
            is_smoke = index == repetition == 1
            if (phase == "smoke") != is_smoke:
                continue
            for group in ("S0", "S1") if (index + repetition) % 2 == 0 else ("S1", "S0"):
                folder = directory / f"{case['public'].task_id}_r{repetition}_{group}"
                if (folder / "result.json").exists():
                    print("Preserved:", folder.name, flush=True)
                    continue
                if folder.exists():
                    raise ValueError("Unfinished episode retained; no automatic paid reissue")
                row = await run_case(
                    case,
                    folder=folder,
                    template=root / ".local/databases/retail_template.sqlite",
                    backend_factory=lambda: DeepSeekBackend(
                        options_from_settings(settings), secret, allow_network=True
                    ),
                    user_factory=lambda script: FormalUser(script, options=options),
                    skill_provider=skill if group == "S1" else None,
                    policy_text=load_application_policy(root),
                    context=RunContext(
                        environment_id="assigned",
                        backend="deepseek",
                        model_name=settings.model_name,
                        temperature=settings.model_temperature,
                        max_output_tokens=settings.model_max_output_tokens,
                        limits=limits,
                        skill_id="after-sales-workflow" if group == "S1" else None,
                        skill_version="1.0.0" if group == "S1" else "none",
                    ),
                    group=group,
                    transport="mcp",
                    protocol=protocol,
                )
                episodes = rows_from(directory)
                save(
                    directory / "summary.json",
                    {
                        "experiment_id": config["experiment_id"],
                        "backend": "deepseek",
                        "selected_split": "dev",
                        "episodes": len(episodes),
                        "expected_episodes": 48,
                        "groups": aggregate([entry["result"] for entry in episodes]),
                        "entries": episodes,
                        "sealed_holdout_live_episodes": 0,
                    },
                )
                print(
                    json.dumps(
                        {
                            "episode": folder.name,
                            "status": row["scoring"]["task_status"],
                            "interaction": row["scoring"]["interaction"]["code"],
                            "calls": row["usage"]["model_calls"],
                            "tokens": row["usage"]["reported_tokens"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                events = [
                    json.loads(s) for s in (folder / "trace.jsonl").read_text("utf-8").splitlines()
                ]
                if (
                    row["usage"]["unknown_usage_calls"] > 0
                    or not row["scoring"]["safety"]["passed"]
                    or row["terminal_status"] == "failed"
                    or any(event["kind"] == "model_error" for event in events)
                ):
                    save(
                        directory / "halt.json",
                        {
                            "episode": folder.name,
                            "reason": "provider/usage/runtime/safety audit required",
                        },
                    )
                    raise RuntimeError("Episode preserved; stopped further paid calls for audit")
    save(directory / "smoke_gate.json", smoke_gate(directory))
    episodes = rows_from(directory)
    if len(episodes) == 48 and not (directory / "language_review").exists():
        export_review(directory, directory / "language_review", protocol)
        save(
            directory / "completed.json",
            {"finished_at_utc": datetime.now(UTC).isoformat(), "episodes": 48},
        )


async def offline_holdout(root: Path, output: Path):
    if output.exists() or not output.resolve().is_relative_to((root / ".local").resolve()):
        raise ValueError("Choose a new offline output directory under .local")
    output.mkdir(parents=True)
    protocol = load_protocol(root)
    entries = []
    for split in ("validation", "test"):
        cases, _ = load_holdout(root, split)
        for case in cases:
            row = await run_case(
                case,
                folder=output / case["public"].task_id,
                template=root / ".local/databases/retail_template.sqlite",
                backend_factory=lambda case=case: ProtocolReplay(case["script"], variant="clarify"),
                policy_text=load_application_policy(root),
                context=RunContext(
                    environment_id="assigned",
                    limits=RunLimits.model_validate_json(
                        (root / "configs/m3_live_limits.json").read_bytes()
                    ),
                ),
                protocol=protocol,
                group="S0",
                transport="native",
            )
            entries.append({"split": split, "result": row})
            save(
                output / "summary.json",
                {
                    "backend": "ProtocolReplay",
                    "episodes": len(entries),
                    "actual_api_calls": 0,
                    "meaning": "Driver/schema audit only; not held-out real Agent performance",
                    "entries": entries,
                },
            )
            print(
                json.dumps({"task": row["task_id"], "status": row["scoring"]["task_status"]}),
                flush=True,
            )
