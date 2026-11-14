"""本阶段编排：开发诊断及至多两个隔离生成请求；不跑留出或S2。"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from aftersalesagent.evaluation.e3_candidates.diagnostics import cases, describe, source_audit
from aftersalesagent.evaluation.e3_candidates.evidence import checked_packet, safe_local, sha
from aftersalesagent.evaluation.e3_candidates.generation import request_for
from aftersalesagent.evaluation.e3_candidates.protocol import PreparationUser, load_protocol
from aftersalesagent.evaluation.e3_v2.replay import ProtocolReplay
from aftersalesagent.evaluation.e3_v2.review import export_review, save
from aftersalesagent.evaluation.e3_v2.runner import run_case
from aftersalesagent.evaluation.e3_v2.scoring import aggregate
from aftersalesagent.local_config import options_from_settings
from aftersalesagent.models.deepseek import DeepSeekBackend
from aftersalesagent.policies.application import load_application_policy
from aftersalesagent.schemas.contracts import ModelResponse, RunContext, RunLimits
from aftersalesagent.skills.base import text_hash
from aftersalesagent.skills.registry import SkillRegistry


def frozen_config(root: Path, settings, evidence: Path):
    reference = json.loads((root / "configs/baselines/e3_v2_s0_s1_dev_v1.json").read_text("utf-8"))
    old = reference["frozen_config"]
    for key in ("dev_files", "sealed_holdout_files"):
        if any(sha(root / name) != digest for name, digest in old[key].items()):
            raise ValueError("Historical development/holdout files changed")
    if sha(root / ".local/databases/retail_template.sqlite") != old["database_template_sha256"]:
        raise ValueError("Database template changed")
    if text_hash(load_application_policy(root)) != old["policy_sha256"]:
        raise ValueError("Core policy changed")
    paths = [
        *root.joinpath("src").rglob("*.py"),
        root / "scripts/run_e3_candidates.py",
        root / "configs/evaluation/e3_protocol_v2_1.json",
        root / "configs/m3_live_limits.json",
        root / "pyproject.toml",
        root / "uv.lock",
        *root.joinpath("skills/after-sales-workflow/1.0.0").glob("*"),
    ]
    packet = checked_packet(evidence)
    request = request_for(root, packet, model=settings.model_name)
    return {
        "stage": "e3-v2.1-diagnostics-and-quarantined-generation-v1",
        "planned_on": "2026-10-28",
        "protocol": load_protocol(root),
        "options": options_from_settings(settings).model_dump(mode="json"),
        "limits": json.loads((root / "configs/m3_live_limits.json").read_text("utf-8")),
        "diagnostics": describe(root),
        "language_prompt_audit": source_audit(root),
        "policy_sha256": old["policy_sha256"],
        "tools_sha256": old["tools_sha256"],
        "database_template_sha256": old["database_template_sha256"],
        "baseline_sha256": sha(root / "configs/baselines/e3_v2_s0_s1_dev_v1.json"),
        "sources": {p.relative_to(root).as_posix(): sha(p) for p in sorted(paths)},
        "historical_guard": {k: old[k] for k in ("dev_files", "sealed_holdout_files")},
        "reviewed_evidence_sha256": sha(evidence / "packet.json"),
        "generation_request": request.model_dump(mode="json"),
        "generation": {
            "maximum_candidates": 2,
            "requests_per_candidate": 1,
            "target": "language_consistency",
            "requires_diagnostic_completion": True,
            "automatic_install_or_release": False,
        },
        "maximum_requests": 146,
        "no_live_holdouts": True,
        "no_S2_evaluation": True,
    }


def entries(directory: Path):
    return [json.loads(p.read_text("utf-8")) for p in sorted(directory.glob("*/result.json"))]


def connectivity_gate(folder: Path):
    result = json.loads((folder / "result.json").read_text("utf-8"))
    events = [json.loads(s) for s in (folder / "trace.jsonl").read_text("utf-8").splitlines()]
    responses = [e["payload"] for e in events if e["kind"] == "model_response"]
    ids = [p["provider_metadata"].get("id") for p in responses]
    ready = (
        result["scoring"]["business"]["checks"]["read_expected_order"]
        and result["usage"]["unknown_usage_calls"] == 0
        and result["usage"]["model_calls"] == len(responses) > 0
        and not any(e["kind"] == "model_error" for e in events)
        and all(ids)
        and len(ids) == len(set(ids))
        and all(p["input_tokens"] is not None and p["output_tokens"] is not None for p in responses)
    )
    return {"ready": bool(ready), "requests": len(responses), "semantic_success_required": False}


async def diagnose(root: Path, *, directory: Path, config, settings, secret, phase: str):
    if phase not in {"smoke", "diagnose"}:
        raise ValueError("No live validation/test/S2 phase")
    directory = safe_local(root, directory)
    run = directory / "diagnostics"
    run.mkdir(parents=True, exist_ok=True)
    saved = run / "config.json"
    if saved.exists() and json.loads(saved.read_text("utf-8")) != config:
        raise ValueError("Run config mismatch")
    if not saved.exists():
        save(saved, config)
        save(run / "run_metadata.json", {"started_at_utc": datetime.now(UTC).isoformat()})
    first_folder = run / "e3diag-language-01_r1"
    if phase == "diagnose" and not connectivity_gate(first_folder)["ready"]:
        raise ValueError("Pass one included live smoke first")
    registry = SkillRegistry(run / "skills.sqlite")
    registry.install(root / "skills", "after-sales-workflow", "1.0.0")
    for case in cases(root):
        for repeat in (1, 2):
            smoke = case["public"].task_id == "e3diag-language-01" and repeat == 1
            if (phase == "smoke") != smoke:
                continue
            folder = run / f"{case['public'].task_id}_r{repeat}"
            if (folder / "result.json").exists():
                continue
            if folder.exists():
                raise ValueError("Unfinished paid episode retained; no automatic retry")
            result = await run_case(
                case,
                folder=folder,
                template=root / ".local/databases/retail_template.sqlite",
                backend_factory=lambda: DeepSeekBackend(
                    options_from_settings(settings), secret, allow_network=True
                ),
                user_factory=PreparationUser,
                skill_provider=registry,
                policy_text=load_application_policy(root),
                protocol=config["protocol"],
                group="S1",
                context=RunContext(
                    environment_id="assigned",
                    backend="deepseek",
                    temperature=settings.model_temperature,
                    max_output_tokens=settings.model_max_output_tokens,
                    model_name=settings.model_name,
                    limits=RunLimits.model_validate(config["limits"]),
                    skill_id="after-sales-workflow",
                    skill_version="1.0.0",
                ),
                transport="mcp",
            )
            rows = entries(run)
            save(run / "summary.json", {"entries": rows, "groups": aggregate(rows)})
            print(
                json.dumps(
                    {
                        "episode": folder.name,
                        "status": result["scoring"]["task_status"],
                        "requests": result["usage"]["model_calls"],
                    }
                ),
                flush=True,
            )
            events = [
                json.loads(line)
                for line in (folder / "trace.jsonl").read_text("utf-8").splitlines()
            ]
            if (
                result["usage"]["unknown_usage_calls"]
                or any(e["kind"] == "model_error" for e in events)
                or not result["scoring"]["safety"]["passed"]
                or result["terminal_status"] == "failed"
            ):
                raise ValueError("Provider/usage/safety issue; stop without retry")
            if smoke:
                gate = connectivity_gate(folder)
                save(run / "smoke_gate.json", gate)
                if not gate["ready"]:
                    raise ValueError("Live smoke gate failed")
    if len(entries(run)) == 12:
        save(
            run / "completed.json",
            {"completed_at_utc": datetime.now(UTC).isoformat(), "episodes": 12},
        )
        if not (run / "language_review").exists():
            export_review(run, run / "language_review", config["protocol"])


class PermissionReplay(ProtocolReplay):
    """可读未来计划的工程夹具：人为加一次准备许可，绝不是自然失败复现。"""

    def __init__(self, script):
        super().__init__(script, variant="direct")
        self.permission_asked = False

    async def generate(self, request):
        response = await super().generate(request)
        if (
            not self.permission_asked
            and response.tool_calls
            and response.tool_calls[0].name == "return_delivered_order_items"
        ):
            self.permission_asked = True
            return ModelResponse(content="Shall I prepare this return request?")
        return response


async def offline(root: Path, directory: Path):
    directory = safe_local(root, directory)
    directory.mkdir(parents=True, exist_ok=False)
    registry = SkillRegistry(directory / "skills.sqlite")
    registry.install(root / "skills", "after-sales-workflow", "1.0.0")
    rows = []
    for case in cases(root):
        row = await run_case(
            case,
            folder=directory / case["public"].task_id,
            template=root / ".local/databases/retail_template.sqlite",
            backend_factory=lambda c=case: PermissionReplay(c["script"]),
            user_factory=PreparationUser,
            skill_provider=registry,
            policy_text=load_application_policy(root),
            protocol=load_protocol(root),
            group="S1",
            context=RunContext(
                environment_id="assigned", skill_id="after-sales-workflow", skill_version="1.0.0"
            ),
            transport="mcp",
        )
        rows.append(row)
    save(
        directory / "summary.json",
        {
            "entries": rows,
            "groups": aggregate(rows),
            "actual_api_calls": 0,
            "warning": "Future-aware fixture, not Agent performance",
        },
    )
    return {
        "episodes": len(rows),
        "success": sum(r["scoring"]["task_status"] == "success" for r in rows),
    }
