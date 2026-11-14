"""E2 live验收入口：默认离线，真实执行要求显式Key文件/付费开关/冻结。"""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from aftersalesagent.business.database import import_pinned_database
from aftersalesagent.config import Settings
from aftersalesagent.evaluation.e2_live_runner import E2Replay, business, knowledge, save, summary
from aftersalesagent.evaluation.e2_live_suite import build, load
from aftersalesagent.evaluation.nl_suite import UserPlan
from aftersalesagent.local_config import load_local_config, options_from_settings
from aftersalesagent.models.deepseek import DeepSeekBackend
from aftersalesagent.policies.application import load_application_policy
from aftersalesagent.schemas.contracts import RunContext, RunLimits

ROOT = Path(__file__).resolve().parents[1]


def freeze_config(settings, manifest, limits, live):
    files = sorted(
        [
            *ROOT.joinpath("src").rglob("*.py"),
            Path(__file__).resolve(),
            ROOT / "uv.lock",
            ROOT / "pyproject.toml",
            ROOT / "configs/m3_live_limits.json",
        ]
    )
    return {
        "suite_id": manifest["suite_id"],
        "task_files": manifest["files"],
        "options": options_from_settings(settings).model_dump(mode="json") if live else None,
        "limits": limits.model_dump(mode="json"),
        "policy_sha256": hashlib.sha256(load_application_policy(ROOT).encode()).hexdigest(),
        "knowledge_manifest_sha256": hashlib.sha256(
            (ROOT / "data/knowledge/e1_v1/manifest.json").read_bytes()
        ).hexdigest(),
        "sources": {
            p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files
        },
        "skill_version": "none",
        "transport_pairs": ["native", "mcp"],
        "fault": "proposal committed then response lost; same operation key, one harness retry",
        "smoke_gate": "known usage and successful order/product tool observations on both routes",
    }


async def execute(args, cases, settings, secret, config):
    root = args.directory.resolve()
    root.mkdir(parents=True, exist_ok=True)
    saved_config = root / "config.json"
    if saved_config.exists():
        if json.loads(saved_config.read_text("utf-8")) != config:
            raise ValueError("Run directory has a different frozen config")
    else:
        save(saved_config, config)
    live = args.backend == "deepseek"
    template = ROOT / ".local/databases/retail_template.sqlite"
    if not template.exists():
        import_pinned_database(ROOT, template).close()
    limits = RunLimits.model_validate(config["limits"])
    selected = [
        c
        for c in cases
        if args.phase == "all"
        or (args.phase == "smoke" and c["public"]["task_id"] == "e2live-01")
        or (args.phase == "remaining" and c["public"]["task_id"] != "e2live-01")
    ]
    if live and args.phase == "remaining":
        gate_path = root / "smoke_gate.json"
        if not gate_path.exists() or not json.loads(gate_path.read_text("utf-8"))["ready"]:
            raise ValueError("Run the frozen Native/MCP smoke pair first")
    for case in selected:
        index = int(case["public"]["task_id"].split("-")[-1])
        for transport in ("native", "mcp") if index % 2 else ("mcp", "native"):
            folder = root / (case["public"]["task_id"] + "_" + transport)
            if (folder / "result.json").exists():
                print("Preserved existing episode:", folder.name, flush=True)
                continue
            if folder.exists():
                raise ValueError("Unfinished episode retained; no automatic paid reissue")
            if case["public"]["kind"] == "knowledge":
                row = await knowledge(
                    case,
                    folder=folder,
                    root=ROOT,
                    template=template,
                    settings=settings,
                    secret=secret,
                    transport=transport,
                    live=live,
                )
            else:
                plan = UserPlan.model_validate(case["user_script"]["plan"])
                factory = (
                    (
                        lambda: DeepSeekBackend(
                            options_from_settings(settings), secret, allow_network=True
                        )
                    )
                    if live
                    else (
                        lambda plan=plan, case=case: E2Replay(
                            plan, case["public"]["scenario"], case["user_script"]["extra"]
                        )
                    )
                )
                context = RunContext(
                    environment_id="assigned-per-episode",
                    backend="deepseek" if live else "mock",
                    model_name=settings.model_name if live else None,
                    limits=limits,
                    temperature=settings.model_temperature,
                    max_output_tokens=settings.model_max_output_tokens,
                )
                row = await business(
                    case,
                    folder=folder,
                    template=template,
                    factory=factory,
                    policy=load_application_policy(ROOT),
                    context=context,
                    transport=transport,
                )
            print(
                json.dumps(
                    {
                        "episode": folder.name,
                        "passed": row["verification"]["success"],
                        "failures": row["verification"]["failure_labels"],
                        "model_requests": row["usage"]["model_calls"],
                        "tokens": row["usage"]["reported_tokens"],
                    }
                ),
                flush=True,
            )
    rows = [json.loads(p.read_text("utf-8")) for p in sorted(root.glob("*/result.json"))]
    smoke = [r for r in rows if r["task_id"] == "e2live-01"]
    if len(smoke) == 2:
        ready = all(
            r["verification"]["checks"].get("read_expected_order")
            and r["verification"]["checks"].get("product_read")
            and (not live or r["usage"]["unknown_usage_calls"] == 0)
            and r["terminal_status"] not in {"failed", "truncated"}
            for r in smoke
        )
        save(
            root / "smoke_gate.json",
            {
                "ready": ready,
                "task_acceptance": [r["verification"]["success"] for r in smoke],
                "note": (
                    "Connectivity/usage gate is separate from all task assertions; retain failures"
                ),
            },
        )
    report = {
        "backend": args.backend,
        "episodes": len(rows),
        "expected_episodes": 16,
        "config": config,
        "metrics": summary(rows),
        "cases": rows,
        "limitations": [
            "Small known-scenario integration check, not population accuracy or a new held-out set",
            "No Skill, no agent tuning, one recorded attempt per task/transport",
            "Knowledge consultation uses deterministic retrieval before model answering",
            "Native and MCP runs can differ because model outputs are not deterministic",
        ],
    }
    save(root / "summary.json", report)
    print("Summary:", root / "summary.json", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--backend", choices=["replay", "deepseek"], default="replay")
    parser.add_argument("--phase", choices=["smoke", "remaining", "all"], default="all")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--directory", type=Path, default=ROOT / ".local/e2_live_runs/replay_v1")
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
        parser.error("Explicit local --env-file is required; never send a key in chat")
    settings, secret = load_local_config(args.env_file) if live else (Settings(), None)
    if live and settings.backend != "deepseek":
        parser.error("The explicit local configuration must select deepseek")
    limits = RunLimits.model_validate_json((ROOT / "configs/m3_live_limits.json").read_bytes())
    config = freeze_config(settings, manifest, limits, live)
    if args.write_freeze:
        if args.execute or args.write_freeze.exists():
            parser.error("Write freeze only in preview; never overwrite")
        args.write_freeze.parent.mkdir(parents=True, exist_ok=True)
        save(args.write_freeze, config)
    if args.freeze_report:
        if json.loads(args.freeze_report.read_text("utf-8")) != config:
            parser.error("Frozen model/source/tasks/limits differ")
    if not args.directory.resolve().is_relative_to((ROOT / ".local").resolve()):
        parser.error("Experiment output must stay in project .local")
    count = 2 if args.phase == "smoke" else 14 if args.phase == "remaining" else 16
    print(
        json.dumps(
            {
                "execute": args.execute,
                "backend": args.backend,
                "phase": args.phase,
                "episodes": count,
                "maximum_model_requests": 24 if count == 2 else 146 if count == 14 else 170,
                "preview_api_calls": 0,
            }
        ),
        flush=True,
    )
    if not args.execute:
        return
    if live and (not args.enable_paid_api or not args.freeze_report or args.phase == "all"):
        parser.error("Live requires paid flag, freeze and staged smoke/remaining execution")
    if live and (secret is None or not secret.get_secret_value()):
        parser.error("Local key is empty")
    asyncio.run(execute(args, cases, settings, secret, config))


if __name__ == "__main__":
    main()
