"""默认只预览；真实调用显式开启，留出集要求已冻结方案。"""

import argparse
import asyncio
import hashlib
import json
import uuid
from pathlib import Path

from aftersalesagent.config import Settings
from aftersalesagent.evaluation.nl_replay import ScriptedNLReplay
from aftersalesagent.evaluation.nl_runner import run_nl_case, summarize
from aftersalesagent.evaluation.nl_suite import json_bytes, load
from aftersalesagent.local_config import load_local_config, options_from_settings
from aftersalesagent.models.deepseek import DeepSeekBackend
from aftersalesagent.policies.application import load_application_policy
from aftersalesagent.schemas.contracts import RunContext, RunLimits

ROOT = Path(__file__).resolve().parents[1]


def fingerprint(settings):
    return hashlib.sha256(options_from_settings(settings).model_dump_json().encode()).hexdigest()


def experiment_config(settings, manifest, policy):
    # 只哈希会影响本次实验的源码/配置；排除.env、运行结果与文档。
    files = sorted(
        [
            *ROOT.joinpath("src").rglob("*.py"),
            ROOT / "scripts/run_nl_evaluation.py",
            ROOT / "configs/m3_live_limits.json",
        ]
    )
    return {
        "model_fingerprint": fingerprint(settings),
        "options": options_from_settings(settings).model_dump(mode="json"),
        "limits": settings.run_limits().model_dump(mode="json"),
        "policy_sha256": hashlib.sha256(policy.encode()).hexdigest(),
        "task_hashes": manifest["files"],
        "source_hashes": {
            str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files
        },
    }


async def execute(args, cases, manifest, settings, secret, policy, config):
    run_id = "nl_" + uuid.uuid4().hex[:10]
    directory = ROOT / ".local/evaluations" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "config.json").write_bytes(json_bytes(config))
    rows = []
    context = RunContext(
        environment_id="per-case",
        backend="deepseek" if args.backend == "deepseek" else "mock",
        model_name=settings.model_name if args.backend == "deepseek" else None,
        temperature=settings.model_temperature,
        max_output_tokens=settings.model_max_output_tokens,
        limits=settings.run_limits(),
    )
    for case in cases:
        factory = (
            (lambda: DeepSeekBackend(options_from_settings(settings), secret, allow_network=True))
            if args.backend == "deepseek"
            else (lambda plan=case.plan: ScriptedNLReplay(plan))
        )
        row = await run_nl_case(
            case,
            template=ROOT / ".local/databases/retail_template.sqlite",
            folder=directory / case.public.task_id,
            backend_factory=factory,
            policy=policy,
            context=context,
        )
        rows.append(row)
        print(
            json.dumps(
                {
                    "task": row["task_id"],
                    "passed": row["verification"]["success"],
                    "failures": row["failure_categories"],
                }
            ),
            flush=True,
        )
    report = {
        "run_id": run_id,
        "suite_id": manifest["suite_id"],
        "backend": args.backend,
        "split": args.split,
        "label": args.label,
        "config": config,
        "metrics": summarize(rows, live=args.backend == "deepseek"),
        "cases": rows,
        "official_tau2_score": False,
        "limitations": manifest["limitations"],
    }
    path = directory / "summary.json"
    path.write_bytes(json_bytes(report))
    print("Summary:", path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["replay", "deepseek"], default="replay")
    parser.add_argument("--split", choices=["dev", "validation", "test"], default="dev")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--label", default="preview")
    parser.add_argument(
        "--task-id", help="Select one task for an explicitly reported development diagnostic"
    )
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--limits-file", type=Path, default=ROOT / "configs/m3_live_limits.json")
    parser.add_argument("--smoke-report", type=Path)
    parser.add_argument("--freeze-report", type=Path)
    parser.add_argument("--write-freeze", type=Path)
    parser.add_argument("--final-test", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--enable-paid-api", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.limit <= 12:
        parser.error("limit must be 1..12")
    cases, manifest = load(ROOT, args.split)
    if args.task_id:
        cases = [case for case in cases if case.public.task_id == args.task_id]
        if len(cases) != 1:
            parser.error("task-id not found in selected split")
    cases = cases[: args.limit]
    settings, secret = (
        load_local_config(args.env_file) if args.backend == "deepseek" else (Settings(), None)
    )
    limits = RunLimits.model_validate_json(args.limits_file.read_bytes())
    settings = settings.model_copy(update=limits.model_dump())
    policy = load_application_policy(ROOT)
    config = (
        experiment_config(settings, manifest, policy)
        if args.backend == "deepseek"
        else {
            "limits": limits.model_dump(mode="json"),
            "task_hashes": manifest["files"],
            "policy_sha256": hashlib.sha256(policy.encode()).hexdigest(),
        }
    )
    if args.write_freeze:
        if args.execute or args.backend != "deepseek":
            parser.error("write-freeze requires deepseek preview and cannot execute")
        if args.write_freeze.exists():
            parser.error("freeze already exists; do not overwrite")
        args.write_freeze.parent.mkdir(parents=True, exist_ok=True)
        args.write_freeze.write_bytes(json_bytes(config))
    if args.freeze_report:
        if json.loads(args.freeze_report.read_text("utf-8")) != config:
            parser.error("Current source/model/tasks/limits differ from frozen experiment")
    print(
        json.dumps(
            {
                "execute": args.execute,
                "backend": args.backend,
                "split": args.split,
                "cases": len(cases),
                "maximum_model_requests": len(cases) * limits.max_model_calls,
                "preview_api_calls": 0,
            },
            indent=2,
        ),
        flush=True,
    )
    if not args.execute:
        return
    if args.split == "test" and not args.final_test:
        parser.error("test requires --final-test")
    if args.backend == "deepseek":
        if not args.enable_paid_api or not args.smoke_report:
            parser.error("Live execution requires explicit paid flag and smoke report")
        smoke = json.loads(args.smoke_report.read_text("utf-8"))
        if not smoke.get("passed") or smoke.get("settings_fingerprint") != fingerprint(settings):
            parser.error("Smoke does not match current model")
        if args.split != "dev" and not args.freeze_report:
            parser.error("Held-out live runs require a frozen experiment")
        if secret is None or not secret.get_secret_value():
            parser.error("Configure key locally")
    asyncio.run(execute(args, cases, manifest, settings, secret, policy, config))


if __name__ == "__main__":
    main()
