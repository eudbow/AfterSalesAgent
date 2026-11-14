"""对原实验08/09/12单列复核：只改变用户可接纳的合法澄清分支。"""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from aftersalesagent.config import Settings
from aftersalesagent.evaluation.e3_dev_runner import aggregate, save
from aftersalesagent.evaluation.e3_dev_suite import load
from aftersalesagent.evaluation.e3_reason_diagnostic import ReasonAskingReplay, run_diagnostic
from aftersalesagent.local_config import load_local_config, options_from_settings
from aftersalesagent.models.deepseek import DeepSeekBackend
from aftersalesagent.policies.application import load_application_policy
from aftersalesagent.schemas.contracts import RunContext, RunLimits

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / ".local/e3_dev_runs/deepseek_20261027_v1"
TASKS = {"e3dev-08", "e3dev-09", "e3dev-12"}


def config(settings, live):
    cases, manifest = load(ROOT)
    parent = json.loads((PARENT / "summary.json").read_text("utf-8")) if live else None
    if live and parent["episodes"] != 48:
        raise ValueError("Finish and preserve the original 48 episodes first")
    parent_calls = sum(r["usage"]["model_calls"] for r in parent["rows"]) if parent else 0
    if parent_calls + 144 > 576:
        raise ValueError(
            "The original overall request ceiling leaves insufficient diagnostic budget"
        )
    files = [
        *ROOT.joinpath("src").rglob("*.py"),
        Path(__file__).resolve(),
        ROOT / "configs/m3_live_limits.json",
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
        *ROOT.joinpath("skills/after-sales-workflow/1.0.0").glob("*"),
    ]
    return cases, {
        "protocol": "e3-reason-clarification-diagnostic-v1",
        "date": "2026-10-27",
        "tasks": sorted(TASKS),
        "repetitions": 2,
        "episodes": 12,
        "selection": (
            "post-hoc diagnostic subset, not an independent benchmark or replacement score"
        ),
        "only_change": "one optional response restating an already expressed cancellation reason",
        "original_task_files": manifest["files"],
        "options": options_from_settings(settings).model_dump(mode="json") if live else None,
        "limits": json.loads((ROOT / "configs/m3_live_limits.json").read_text("utf-8")),
        "parent_completed_calls": parent_calls,
        "maximum_additional_requests": 144,
        "overall_request_ceiling": 576,
        "groups": {"S0": "none", "S1": "after-sales-workflow@1.0.0"},
        "sources": {
            p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(files)
            if p.is_file()
        },
    }


async def execute(args, cases, settings, secret, frozen):
    output = args.directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    stored = output / "config.json"
    if stored.exists():
        if json.loads(stored.read_text("utf-8")) != frozen:
            raise ValueError("Diagnostic config mismatch")
    else:
        save(stored, frozen)
    if args.backend == "deepseek" and args.phase == "remaining":
        if not json.loads((output / "smoke_gate.json").read_text("utf-8"))["ready"]:
            raise ValueError("Diagnostic smoke must pass first")
    for case in cases:
        task = case["public"]["task_id"]
        if task not in TASKS:
            continue
        index = int(task.split("-")[-1])
        for repetition in (1, 2):
            smoke = task == "e3dev-08" and repetition == 1
            if args.phase == "smoke" and not smoke or args.phase == "remaining" and smoke:
                continue
            for group in ("S0", "S1") if (index + repetition) % 2 == 0 else ("S1", "S0"):
                folder = output / f"{task}_r{repetition}_{group}"
                if (folder / "result.json").exists():
                    print("Preserved:", folder.name, flush=True)
                    continue
                if folder.exists():
                    raise ValueError("Unfinished diagnostic retained; automatic reissue refused")
                factory = (
                    (
                        lambda: DeepSeekBackend(
                            options_from_settings(settings), secret, allow_network=True
                        )
                    )
                    if args.backend == "deepseek"
                    else lambda case=case: ReasonAskingReplay(case["plan"])
                )
                row = await run_diagnostic(
                    case,
                    folder=folder,
                    template=ROOT / ".local/databases/retail_template.sqlite",
                    factory=factory,
                    policy=load_application_policy(ROOT),
                    context=RunContext(
                        environment_id="assigned",
                        backend="deepseek" if args.backend == "deepseek" else "mock",
                        model_name=settings.model_name if args.backend == "deepseek" else None,
                        limits=RunLimits.model_validate(frozen["limits"]),
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
                rows = [
                    json.loads(p.read_text("utf-8")) for p in sorted(output.glob("*/result.json"))
                ]
                save(
                    output / "summary.json",
                    {
                        "protocol": frozen["protocol"],
                        "backend": args.backend,
                        "episodes": len(rows),
                        "expected_episodes": 12,
                        "metrics": aggregate(rows),
                        "rows": rows,
                        "frozen_config": frozen,
                    },
                )
    smoke_rows = [
        json.loads(p.read_text("utf-8")) for p in output.glob("e3dev-08_r1_*/result.json")
    ]
    if len(smoke_rows) == 2:
        save(
            output / "smoke_gate.json",
            {
                "ready": all(
                    r["verification"]["checks"]["read_expected_order"]
                    and (args.backend != "deepseek" or r["usage"]["unknown_usage_calls"] == 0)
                    for r in smoke_rows
                ),
                "task_success": [r["verification"]["success"] for r in smoke_rows],
            },
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["replay", "deepseek"], default="replay")
    parser.add_argument("--phase", choices=["smoke", "remaining", "all"], default="all")
    parser.add_argument("--directory", type=Path, default=ROOT / ".local/e3_reason_runs/replay_v1")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--write-freeze", type=Path)
    parser.add_argument("--freeze-report", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--enable-paid-api", action="store_true")
    args = parser.parse_args()
    live = args.backend == "deepseek"
    if live and not args.env_file:
        parser.error("Explicit local env file required")
    settings, secret = load_local_config(args.env_file) if live else (Settings(), None)
    if live and (settings.backend != "deepseek" or not secret or not secret.get_secret_value()):
        parser.error("DeepSeek local config missing")
    cases, frozen = config(settings, live)
    if args.write_freeze:
        if args.execute or args.write_freeze.exists():
            parser.error("Freeze must be written in preview without overwriting")
        args.write_freeze.parent.mkdir(parents=True, exist_ok=True)
        save(args.write_freeze, frozen)
    if args.freeze_report and json.loads(args.freeze_report.read_text("utf-8")) != frozen:
        parser.error("Diagnostic frozen config differs")
    if not args.directory.resolve().is_relative_to((ROOT / ".local").resolve()):
        parser.error("Output must stay under project .local")
    print(
        json.dumps(
            {
                "backend": args.backend,
                "execute": args.execute,
                "episodes": 12,
                "maximum_additional_requests": 144,
                "preview_api_calls": 0,
            }
        ),
        flush=True,
    )
    if not args.execute:
        return
    if live and (not args.enable_paid_api or not args.freeze_report or args.phase == "all"):
        parser.error("Live needs paid flag, freeze and staged execution")
    asyncio.run(execute(args, cases, settings, secret, frozen))


if __name__ == "__main__":
    main()
