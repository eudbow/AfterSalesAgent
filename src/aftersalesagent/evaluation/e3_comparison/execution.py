"""新跑的S1/S2成对比较：固定输入、显式付费、按阶段门槛解锁，旧数据永不重判。"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

from aftersalesagent.business.database import RetailStore
from aftersalesagent.business.service import RetailService
from aftersalesagent.evaluation.e3_candidates.evidence import safe_local
from aftersalesagent.evaluation.e3_candidates.protocol import PreparationUser, load_protocol
from aftersalesagent.evaluation.e3_comparison.gates import evaluate
from aftersalesagent.evaluation.e3_comparison.materials import MANIFEST, load, planned_pairs
from aftersalesagent.evaluation.e3_comparison.provider import MonitoredBackend
from aftersalesagent.evaluation.e3_comparison.review import evidence, export_review
from aftersalesagent.evaluation.e3_v2.replay import ProtocolReplay
from aftersalesagent.evaluation.e3_v2.review import save
from aftersalesagent.evaluation.e3_v2.runner import run_case
from aftersalesagent.evaluation.e3_v2.suite import sha
from aftersalesagent.local_config import options_from_settings
from aftersalesagent.models.deepseek import DeepSeekBackend
from aftersalesagent.policies.application import load_application_policy
from aftersalesagent.schemas.contracts import RunContext, RunLimits
from aftersalesagent.skills.base import text_hash, tools_hash
from aftersalesagent.skills.registry import SkillRegistry

RULES = "configs/evaluation/e3_skill_selection_v1.json"
CANDIDATE = ".local/e3_candidate_step/20261028_v1/generation/1.0.1/bundle"
CLAIM_ALERT = re.compile(
    r"\b(?:I have|I've|we have|we've) (?:successfully )?(?:cancelled|canceled|refunded|updated)\b|"
    r"\b(?:refund|order|return request) (?:has been|is now) "
    r"(?:completed|issued|cancelled|canceled)\b|"
    r"(?:已经|已)(?:成功)?(?:取消订单|更新地址|完成退款|退款到账|执行了)",
    re.I,
)


def registry(root: Path, path: Path):
    reg = SkillRegistry(path)
    for group, bundle in (("S1", "skills"), ("S2", CANDIDATE)):
        version = "1.0.0" if group == "S1" else "1.0.1"
        content = reg.install(root / bundle, "after-sales-workflow", version)
        rules = json.loads((root / RULES).read_text("utf-8"))
        if content.content_sha256 != rules["groups"][group]["content_sha256"]:
            raise ValueError("Skill bytes differ from frozen rules")
    return reg


def freeze(root: Path, settings):
    rules = json.loads((root / RULES).read_text("utf-8"))
    options = options_from_settings(settings).model_dump(mode="json")
    comparison = rules["comparison"]
    if (
        settings.backend != "deepseek"
        or options["model"] != comparison["model"]
        or options["temperature"] != 0
        or options["max_output_tokens"] != 1024
    ):
        raise ValueError("Local model settings differ from predeclared rules")
    manifest = json.loads((root / MANIFEST).read_text("utf-8"))
    for split in manifest["groups"]:
        load(root, split)
    paths = [
        *root.joinpath("src").rglob("*.py"),
        root / "scripts/run_e3_comparison.py",
        root / RULES,
        root / "configs/evaluation/e3_protocol_v2_1.json",
        root / "configs/m3_live_limits.json",
        root / "pyproject.toml",
        root / "uv.lock",
        *root.joinpath("skills/after-sales-workflow/1.0.0").glob("*"),
        *root.joinpath(CANDIDATE, "after-sales-workflow/1.0.1").glob("*"),
    ]
    template = root / ".local/databases/retail_template.sqlite"
    store = RetailStore(template)
    try:
        schema = tools_hash(RetailService(store).specs())
    finally:
        store.close()
    previous = json.loads((root / "configs/baselines/e3_v2_s0_s1_dev_v1.json").read_text("utf-8"))
    policy = text_hash(load_application_policy(root))
    if (
        schema != previous["frozen_config"]["tools_sha256"]
        or policy != previous["frozen_config"]["policy_sha256"]
    ):
        raise ValueError("Core tools/policy changed")
    return {
        "experiment_id": "e3-S1-S2-matched-v1",
        "prepared_on": "2026-10-28",
        "options": options,
        "rules": rules,
        "protocol": load_protocol(root),
        "limits": rules["limits"],
        "material_manifest_sha256": sha(root / MANIFEST),
        "material_files": manifest["files"],
        "material_source_files": manifest["source_files"],
        "policy_sha256": policy,
        "tools_sha256": schema,
        "template_sha256": sha(template),
        "sources": {p.relative_to(root).as_posix(): sha(p) for p in sorted(paths)},
        "stages": rules["stages"],
        "automatic_release": False,
    }


def planned_episode_ids(root: Path, split: str):
    return [
        f"{task}_r{repeat}_{group}"
        for task, repeat in planned_pairs(root, split)
        for group in ("S1", "S2")
    ]


def audit_pairs(root: Path, directory: Path, split: str):
    rules = json.loads((root / RULES).read_text("utf-8"))
    issues = []
    for task, repeat in planned_pairs(root, split):
        inputs = []
        for group in ("S1", "S2"):
            folder = directory / split / f"{task}_r{repeat}_{group}"
            if not (folder / "result.json").exists():
                continue
            events = [
                json.loads(s) for s in (folder / "trace.jsonl").read_text("utf-8").splitlines()
            ]
            request = next((e["payload"] for e in events if e["kind"] == "model_request"), None)
            if request is None:
                issues.append("missing_initial_request")
                continue
            expected = rules["groups"][group]
            if (
                request["skill_version"] != expected["version"]
                or request["skill_sha256"] != expected["content_sha256"]
            ):
                issues.append("actual_skill_snapshot_mismatch")
            inputs.append(
                {
                    "core": request["messages"][0],
                    "tools": request["tools"],
                    "users": [m for m in request["messages"] if m["role"] == "user"],
                    "model": request["model_name"],
                    "temperature": request["temperature"],
                    "max_tokens": request["max_output_tokens"],
                    "initial_state_sha256": sha(folder / "initial_state.json"),
                }
            )
        if len(inputs) == 2 and inputs[0] != inputs[1]:
            issues.append("paired_initial_conditions_differ")
    return issues


def decide(root: Path, directory: Path, split: str):
    rules = json.loads((root / RULES).read_text("utf-8"))
    cases, _ = load(root, split)
    folder = directory / split
    episodes, primary, complete = evidence(folder, cases)
    technical = audit_pairs(root, directory, split)
    adjudication = folder / "review/adjudication.json"
    if adjudication.exists() and json.loads(adjudication.read_text("utf-8"))[
        "reviewer_kind"
    ] not in {"human", "assistant"}:
        technical.append("invalid_live_review_provenance")
    ledger = directory / "provider_ledger.jsonl"
    if ledger.exists():
        technical.extend(
            row["error"]
            for row in (json.loads(s) for s in ledger.read_text("utf-8").splitlines())
            if row["error"]
        )
    verdict = evaluate(
        rules,
        split,
        planned_pairs(root, split),
        episodes,
        primary,
        technical=technical,
        review_complete=complete,
    )
    sources = {p.relative_to(directory).as_posix(): sha(p) for p in folder.glob("*/result.json")}
    review = folder / "review/adjudication.json"
    if review.exists():
        sources[review.relative_to(directory).as_posix()] = sha(review)
    verdict["sources"] = sources
    verdict["primary_pairs"] = primary
    return verdict


def save_decision(root: Path, directory: Path, split: str):
    verdict = decide(root, directory, split)
    target = directory / split / "decision.json"
    if target.exists():
        if json.loads(target.read_text("utf-8")) != verdict:
            raise ValueError("Never rewrite a stage decision")
    else:
        save(target, verdict)
    return verdict


def unlocked(root: Path, directory: Path, split: str):
    previous = {"validation": "development", "test": "validation"}.get(split)
    if previous:
        recorded = directory / previous / "decision.json"
        if not recorded.exists() or decide(root, directory, previous) != json.loads(
            recorded.read_text("utf-8")
        ):
            raise ValueError("Previous stage decision/source binding missing or changed")
        if json.loads(recorded.read_text("utf-8"))["status"] != "passed":
            raise ValueError("Previous stage did not pass; holdout stays sealed")


def smoke_gate(directory: Path, first_id: str):
    ready = True
    ids = []
    for group in ("S1", "S2"):
        path = directory / "development" / f"{first_id}_r1_{group}"
        if not (path / "result.json").exists():
            ready = False
            continue
        row = json.loads((path / "result.json").read_text("utf-8"))
        events = [json.loads(s) for s in (path / "trace.jsonl").read_text("utf-8").splitlines()]
        responses = [e["payload"] for e in events if e["kind"] == "model_response"]
        ids += [r["provider_metadata"].get("id") for r in responses]
        ready &= (
            row["scoring"]["business"]["checks"]["located_expected_user"]
            and row["scoring"]["business"]["checks"]["read_expected_order"]
            and row["usage"]["unknown_usage_calls"] == 0
            and row["usage"]["model_calls"] == len(responses) > 0
            and not any(e["kind"] == "model_error" for e in events)
        )
    return {
        "ready": bool(ready and all(ids) and len(ids) == len(set(ids)) and ids),
        "response_count": len(ids),
        "semantic_score_filter": False,
    }


async def live(root: Path, directory: Path, split: str, phase: str, settings, secret, config):
    if phase not in {"smoke", "remaining"} or split not in {"development", "validation", "test"}:
        raise ValueError("Unknown comparison phase")
    if split != "development" and phase == "smoke":
        raise ValueError("Only development has the included two-episode smoke")
    directory = safe_local(root, directory)
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / "config.json"
    if config_path.exists() and json.loads(config_path.read_text("utf-8")) != config:
        raise ValueError("Live run config mismatch")
    if not config_path.exists():
        save(config_path, config)
        save(directory / "run_metadata.json", {"started_at_utc": datetime.now(UTC).isoformat()})
    unlocked(root, directory, split)
    cases, _ = load(root, split)
    stage = directory / split
    stage.mkdir(exist_ok=True)
    if (stage / "halt.json").exists():
        raise ValueError("Halted stage cannot automatically resume")
    committed_path = stage / "committed.json"
    committed = json.loads(committed_path.read_text("utf-8")) if committed_path.exists() else {}
    for name, files in committed.items():
        if any(sha(stage / name / file) != digest for file, digest in files.items()):
            raise ValueError("Committed episode artifacts changed")
    if split == "development" and phase == "remaining":
        if not smoke_gate(directory, cases[0]["public"].task_id)["ready"]:
            raise ValueError("Pass two included real smoke episodes first")
    if split == "test" and not (stage / "candidate_final_freeze.json").exists():
        save(
            stage / "candidate_final_freeze.json",
            {
                "config_sha256": sha(config_path),
                "validation_decision_sha256": sha(directory / "validation/decision.json"),
                "candidate": config["rules"]["groups"]["S2"],
            },
        )
    reg = registry(root, directory / "experimental_skills.sqlite")
    for index, case in enumerate(cases):
        for repeat in (1, 2):
            is_smoke = split == "development" and index == 0 and repeat == 1
            if split == "development" and (phase == "smoke") != is_smoke:
                continue
            order = ("S1", "S2") if (index + 1 + repeat) % 2 == 0 else ("S2", "S1")
            for group in order:
                episode = f"{case['public'].task_id}_r{repeat}_{group}"
                folder = stage / episode
                if episode in committed:
                    continue
                if folder.exists():
                    raise ValueError("Unfinished paid episode retained; no automatic replay")

                def backend(eid=episode):
                    return MonitoredBackend(
                        DeepSeekBackend(
                            options_from_settings(settings), secret, allow_network=True
                        ),
                        directory / "provider_ledger.jsonl",
                        eid,
                        settings.model_name,
                    )

                row = await run_case(
                    case,
                    folder=folder,
                    template=root / ".local/databases/retail_template.sqlite",
                    backend_factory=backend,
                    user_factory=PreparationUser,
                    skill_provider=reg,
                    policy_text=load_application_policy(root),
                    protocol=config["protocol"],
                    context=RunContext(
                        environment_id="assigned",
                        backend="deepseek",
                        model_name=settings.model_name,
                        temperature=settings.model_temperature,
                        max_output_tokens=settings.model_max_output_tokens,
                        limits=RunLimits.model_validate(config["limits"]),
                        skill_id="after-sales-workflow",
                        skill_version=config["rules"]["groups"][group]["version"],
                    ),
                    group=group,
                    transport="mcp",
                )
                committed[episode] = {
                    p.name: sha(p) for p in folder.iterdir() if p.suffix in {".json", ".jsonl"}
                }
                save(committed_path, committed)
                summary = [
                    {
                        "episode_id": name,
                        "result": json.loads((stage / name / "result.json").read_text("utf-8")),
                    }
                    for name in sorted(committed)
                ]
                save(stage / "summary.json", {"entries": summary})
                print(
                    json.dumps(
                        {
                            "episode": episode,
                            "status": row["scoring"]["task_status"],
                            "requests": row["usage"]["model_calls"],
                        }
                    ),
                    flush=True,
                )
                events = [
                    json.loads(s) for s in (folder / "trace.jsonl").read_text("utf-8").splitlines()
                ]
                technical = [
                    e["payload"].get("code", "model_error")
                    for e in events
                    if e["kind"] == "model_error"
                ]
                alerts = [
                    r
                    for r in row["scoring"]["language"]["records"]
                    if CLAIM_ALERT.search(r["text"])
                ]
                if (
                    technical
                    or row["usage"]["unknown_usage_calls"]
                    or not row["scoring"]["safety"]["passed"]
                    or alerts
                ):
                    save(
                        stage / "halt.json",
                        {
                            "episode_id": episode,
                            "technical": technical,
                            "safe": row["scoring"]["safety"]["passed"],
                            "critical_claim_review_required": bool(alerts),
                            "status": "halted_pending_audit_no_automatic_replay",
                        },
                    )
                    export_review(stage, cases)
                    return {"status": "halted", "episode": episode}
    if split == "development" and phase == "smoke":
        gate = smoke_gate(directory, cases[0]["public"].task_id)
        save(stage / "smoke_gate.json", gate)
        return gate
    expected = planned_episode_ids(root, split)
    if set(committed) != set(expected):
        raise ValueError("Stage unexpectedly incomplete")
    save(
        stage / "completed.json",
        {"episodes": len(expected), "completed_at_utc": datetime.now(UTC).isoformat()},
    )
    if not (stage / "review").exists():
        export_review(stage, cases)
    return {"status": "completed_pending_blind_review", "episodes": len(expected)}


async def offline(root: Path, directory: Path):
    directory = safe_local(root, directory)
    directory.mkdir(parents=True, exist_ok=False)
    reg = registry(root, directory / "experimental_skills.sqlite")
    rows = []
    for split in ("development", "validation", "test"):
        for case in load(root, split)[0]:
            row = await run_case(
                case,
                folder=directory / split / case["public"].task_id,
                template=root / ".local/databases/retail_template.sqlite",
                backend_factory=lambda c=case: ProtocolReplay(c["script"], variant="direct"),
                user_factory=PreparationUser,
                skill_provider=reg,
                context=RunContext(
                    environment_id="assigned",
                    skill_id="after-sales-workflow",
                    skill_version="1.0.0",
                ),
                policy_text=load_application_policy(root),
                protocol=load_protocol(root),
                group="S1",
                transport="native",
            )
            rows.append({"split": split, "result": row})
            print(
                json.dumps(
                    {"offline": case["public"].task_id, "status": row["scoring"]["task_status"]}
                ),
                flush=True,
            )
    save(
        directory / "summary.json",
        {
            "entries": rows,
            "actual_api_calls": 0,
            "meaning": "Future-aware driver audit only; not heldout model scores",
        },
    )
    return {
        "tasks": len(rows),
        "success": sum(r["result"]["scoring"]["task_status"] == "success" for r in rows),
    }
