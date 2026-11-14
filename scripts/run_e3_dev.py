"""E3 新开发 S0/S1 配对实验；默认预览，付费执行须冻结及冒烟门槛。"""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from aftersalesagent.business.database import import_pinned_database
from aftersalesagent.config import Settings
from aftersalesagent.evaluation.e3_dev_runner import WorkflowReplay, aggregate, run_case, save
from aftersalesagent.evaluation.e3_dev_suite import build, load
from aftersalesagent.local_config import load_local_config, options_from_settings
from aftersalesagent.models.deepseek import DeepSeekBackend
from aftersalesagent.policies.application import load_application_policy
from aftersalesagent.schemas.contracts import RunContext, RunLimits

ROOT = Path(__file__).resolve().parents[1]


def frozen_config(settings, manifest, live):
    source = sorted(
        [
            *ROOT.joinpath("src").rglob("*.py"),
            Path(__file__).resolve(),
            ROOT / "pyproject.toml",
            ROOT / "uv.lock",
            ROOT / "configs/m3_live_limits.json",
            *ROOT.joinpath("skills/after-sales-workflow/1.0.0").glob("*"),
        ]
    )
    return {
        "suite_id": manifest["suite_id"],
        "task_files": manifest["files"],
        "manifest_sha256": hashlib.sha256(
            (ROOT / "data/manifests/e3_dev_v1_suite.json").read_bytes()
        ).hexdigest(),
        "backend": "deepseek" if live else "replay",
        "options": options_from_settings(settings).model_dump(mode="json") if live else None,
        "limits": json.loads((ROOT / "configs/m3_live_limits.json").read_text("utf-8")),
        "policy_sha256": hashlib.sha256(load_application_policy(ROOT).encode()).hexdigest(),
        "transport": "mcp",
        "groups": {
            "S0": {"skill_id": None, "skill_version": "none"},
            "S1": {"skill_id": "after-sales-workflow", "skill_version": "1.0.0"},
        },
        "repetitions": 2,
        "expected_episodes": 48,
        "maximum_model_requests": 576,
        "primary": manifest["primary"],
        "secondary": manifest["secondary"],
        "selection": "development diagnostic only; no candidate optimization in this run",
        "sources": {
            p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source
            if p.is_file()
        },
        "smoke_gate": (
            "successful authenticated order read in both groups; known usage; no provider errors"
        ),
    }


def episode_id(task, rep, group):
    return f"{task}_r{rep}_{group}"


async def execute(args, cases, settings, secret, config):
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "config.json"
    if path.exists():
        if json.loads(path.read_text("utf-8")) != config:
            raise ValueError("Existing experiment has a different frozen config")
    else:
        save(path, config)
    live = args.backend == "deepseek"
    if live and args.phase == "remaining":
        gate = directory / "smoke_gate.json"
        if not gate.exists() or not json.loads(gate.read_text("utf-8"))["ready"]:
            raise ValueError("Complete the two-episode smoke gate first")
    template = ROOT / ".local/databases/retail_template.sqlite"
    if not template.exists():
        import_pinned_database(ROOT, template).close()
    limits = RunLimits.model_validate(config["limits"])
    for i, case in enumerate(cases, 1):
        for repetition in (1, 2):
            smoke = i == 1 and repetition == 1
            if args.phase == "smoke" and not smoke or args.phase == "remaining" and smoke:
                continue
            for group in ("S0", "S1") if (i + repetition) % 2 == 0 else ("S1", "S0"):
                folder = directory / episode_id(case["public"]["task_id"], repetition, group)
                if (folder / "result.json").exists():
                    print("Preserved:", folder.name, flush=True)
                    continue
                if folder.exists():
                    raise ValueError("Unfinished episode retained; automatic paid reissue refused")
                factory = (
                    (
                        lambda: DeepSeekBackend(
                            options_from_settings(settings), secret, allow_network=True
                        )
                    )
                    if live
                    else lambda case=case: WorkflowReplay(case["plan"])
                )
                row = await run_case(
                    case,
                    folder=folder,
                    template=template,
                    factory=factory,
                    policy=load_application_policy(ROOT),
                    context=RunContext(
                        environment_id="assigned-per-episode",
                        backend="deepseek" if live else "mock",
                        model_name=settings.model_name if live else None,
                        limits=limits,
                        temperature=settings.model_temperature,
                        max_output_tokens=settings.model_max_output_tokens,
                    ),
                    group=group,
                    repetition=repetition,
                    root=ROOT,
                )
                print(
                    json.dumps(
                        {
                            "episode": folder.name,
                            "passed": row["verification"]["success"],
                            "failures": row["verification"]["failure_labels"],
                            "calls": row["usage"]["model_calls"],
                            "tokens": row["usage"]["reported_tokens"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                # 每个 episode 后更新汇总，即使后续中断也保留已经结束的结果。
                rows = [
                    json.loads(p.read_text("utf-8"))
                    for p in sorted(directory.glob("*/result.json"))
                ]
                save(
                    directory / "summary.json",
                    {
                        "backend": args.backend,
                        "episodes": len(rows),
                        "expected_episodes": 48,
                        "metrics": aggregate(rows),
                        "rows": rows,
                        "frozen_config": config,
                    },
                )
    smoke_rows = [
        json.loads((directory / episode_id("e3dev-01", 1, g) / "result.json").read_text("utf-8"))
        for g in ("S0", "S1")
        if (directory / episode_id("e3dev-01", 1, g) / "result.json").exists()
    ]
    if len(smoke_rows) == 2:
        ready = all(
            r["verification"]["checks"]["read_expected_order"]
            and (
                not live
                or (
                    r["usage"]["unknown_usage_calls"] == 0
                    and r["usage"]["model_calls"] > 0
                    and r["termination_reason"]
                    not in {
                        "model_backend_error",
                        "model_configuration_mismatch",
                        "authentication_failed",
                        "infrastructure_error",
                    }
                )
            )
            for r in smoke_rows
        )
        save(
            directory / "smoke_gate.json",
            {
                "ready": ready,
                "task_passed": [r["verification"]["success"] for r in smoke_rows],
                "note": (
                    "Connectivity/usage gate, not a requirement that all semantic assertions pass"
                ),
            },
        )
    print("Saved:", directory / "summary.json", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--backend", choices=["replay", "deepseek"], default="replay")
    parser.add_argument("--phase", choices=["all", "smoke", "remaining"], default="all")
    parser.add_argument("--directory", type=Path, default=ROOT / ".local/e3_dev_runs/replay_v1")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--write-freeze", type=Path)
    parser.add_argument("--freeze-report", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--enable-paid-api", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        build(ROOT)
    cases, manifest = load(ROOT)
    live = args.backend == "deepseek"
    if live and args.env_file is None:
        parser.error("Live mode requires an explicit local env file")
    settings, secret = load_local_config(args.env_file) if live else (Settings(), None)
    if live and (settings.backend != "deepseek" or not secret or not secret.get_secret_value()):
        parser.error("Local DeepSeek configuration is missing")
    config = frozen_config(settings, manifest, live)
    if args.write_freeze:
        if args.execute or args.write_freeze.exists():
            parser.error("Freeze must be written in preview, never overwritten")
        args.write_freeze.parent.mkdir(parents=True, exist_ok=True)
        save(args.write_freeze, config)
    if args.freeze_report and json.loads(args.freeze_report.read_text("utf-8")) != config:
        parser.error("Frozen source/tasks/model/Skill/limits differ")
    if not args.directory.resolve().is_relative_to((ROOT / ".local").resolve()):
        parser.error("Run output must stay under project .local")
    print(
        json.dumps(
            {
                "backend": args.backend,
                "execute": args.execute,
                "phase": args.phase,
                "total_episodes": 48,
                "maximum_model_requests": 576,
                "preview_api_calls": 0,
            }
        ),
        flush=True,
    )
    if not args.execute:
        return
    if live and (not args.enable_paid_api or not args.freeze_report or args.phase == "all"):
        parser.error("Live execution needs paid opt-in, matching freeze and staged smoke/remaining")
    asyncio.run(execute(args, cases, settings, secret, config))


if __name__ == "__main__":
    main()
