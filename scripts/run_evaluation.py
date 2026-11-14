"""默认仅预览计划；replay离线工程回归，真实评测必须显式启用并提供成功冒烟报告。"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import uuid
from pathlib import Path

from aftersalesagent.business.database import import_pinned_database
from aftersalesagent.config import Settings
from aftersalesagent.evaluation.pricing import PriceSchedule
from aftersalesagent.evaluation.replay import PublicRequestReplay
from aftersalesagent.evaluation.runner import run_case, summarize
from aftersalesagent.evaluation.suite import KINDS, load_cases
from aftersalesagent.local_config import load_local_config, options_from_settings
from aftersalesagent.models.deepseek import DeepSeekBackend
from aftersalesagent.policies.application import load_application_policy
from aftersalesagent.schemas.contracts import RunContext, RunLimits

ROOT = Path(__file__).resolve().parents[1]


async def execute(args, cases, manifest, settings, secret):
    run_id = "m3_" + uuid.uuid4().hex[:10]
    directory = ROOT / ".local/evaluations" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    template = ROOT / ".local/databases/retail_template.sqlite"
    if not template.exists():
        store = import_pinned_database(ROOT, template)
        store.close()
    policy = load_application_policy(ROOT)
    options = options_from_settings(settings) if args.backend == "deepseek" else None
    context = RunContext.model_validate(
        {
            "environment_id": "selected-per-case",
            "backend": "deepseek" if options else "mock",
            "model_name": options.model if options else None,
            "limits": settings.run_limits(),
            "temperature": settings.model_temperature,
            "max_output_tokens": settings.model_max_output_tokens,
        }
    )
    baselines = ["B0", "B1"] if args.baseline == "both" else [args.baseline]
    pricing = (
        PriceSchedule.model_validate_json(args.pricing_file.read_bytes())
        if args.pricing_file
        else None
    )
    rows = []
    for case in cases:
        for baseline in baselines:
            factory = (
                (lambda: DeepSeekBackend(options, secret, allow_network=True))
                if options
                else PublicRequestReplay
            )
            row = await run_case(
                case,
                template=template,
                folder=directory / (case.public.task_id + "_" + baseline),
                baseline=baseline,
                backend_factory=factory,
                policy=policy,
                context=context,
                profile=args.profile,
                pricing=pricing,
            )
            rows.append(row)
            print(
                json.dumps(
                    {
                        "task": row["task_id"],
                        "baseline": baseline,
                        "checks_passed": row["verification"]["success"],
                    }
                )
            )
    config = {
        "backend": args.backend,
        "split": args.split,
        "profile": args.profile,
        "context": context.model_dump(mode="json"),
        "settings": settings.model_dump(mode="json"),
        "policy_sha256": hashlib.sha256(policy.encode()).hexdigest(),
        "smoke_report": str(args.smoke_report) if args.smoke_report else None,
        "limits_file": str(args.limits_file) if args.limits_file else None,
        "limits_file_sha256": hashlib.sha256(args.limits_file.read_bytes()).hexdigest()
        if args.limits_file
        else None,
    }
    report = summarize(rows, backend=args.backend, manifest=manifest, config=config)
    report["run_id"] = run_id
    report["cases"] = rows
    path = directory / "summary.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Summary:", path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["replay", "deepseek"], default="replay")
    parser.add_argument("--baseline", choices=["B0", "B1", "both"], default="both")
    parser.add_argument("--split", choices=["dev", "validation", "test"], default="dev")
    parser.add_argument(
        "--profile",
        choices=["normal", "restart_before_confirm", "commit_response_loss"],
        default="normal",
    )
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--enable-paid-api", action="store_true")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--smoke-report", type=Path)
    parser.add_argument("--pricing-file", type=Path)
    parser.add_argument(
        "--limits-file", type=Path, help="Public RunLimits JSON; never contains credentials"
    )
    parser.add_argument("--final-test", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.limit <= 54:
        parser.error("--limit must be between 1 and 54")
    cases, manifest = load_cases(ROOT, args.split)
    cases = sorted(
        cases, key=lambda c: (int(c.public.task_id.rsplit("-", 1)[1]), KINDS.index(c.criteria.kind))
    )
    cases = cases[: args.limit]
    settings, secret = (
        load_local_config(args.env_file) if args.backend == "deepseek" else (Settings(), None)
    )
    # 评测预算与密钥文件分开；RunLimits严格拒绝未知字段，双方使用同一份配置。
    if args.limits_file is not None:
        limits = RunLimits.model_validate_json(args.limits_file.read_bytes())
        settings = settings.model_copy(update=limits.model_dump())
    plan = {
        "mode": "execution_plan",
        "execute": args.execute,
        "backend": args.backend,
        "cases": len(cases),
        "baselines": 2 if args.baseline == "both" else 1,
        "split": args.split,
        "profile": args.profile,
        "limits": settings.run_limits().model_dump(mode="json"),
        "maximum_model_requests": len(cases)
        * (2 if args.baseline == "both" else 1)
        * settings.max_model_calls,
        "real_api_requests_in_preview": 0,
        "task_success_rate": None,
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if not args.execute:
        return
    if args.split == "test" and not args.final_test:
        parser.error("Final test requires --final-test; do not tune on test results")
    if args.backend == "deepseek":
        if not args.enable_paid_api or args.smoke_report is None:
            parser.error("Live batch requires --enable-paid-api and a successful --smoke-report")
        options = options_from_settings(settings)
        report = json.loads(args.smoke_report.read_text("utf-8"))
        current = hashlib.sha256(options.model_dump_json().encode()).hexdigest()
        if not (
            report.get("mode") == "live_tool_smoke"
            and report.get("passed") is True
            and report.get("settings_fingerprint") == current
        ):
            parser.error("Smoke report does not match the current model configuration")
        if secret is None or not secret.get_secret_value():
            parser.error("Configure the API key locally")
    asyncio.run(execute(args, cases, manifest, settings, secret))


if __name__ == "__main__":
    main()
