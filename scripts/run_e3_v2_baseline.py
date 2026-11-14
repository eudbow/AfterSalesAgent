"""补齐v2留出材料并建立开发集真实S0/S1基线；默认命令不发API请求。"""

import argparse
import asyncio
import json
from pathlib import Path

from aftersalesagent.evaluation.e3_v2.baseline import freeze_config, live, offline_holdout
from aftersalesagent.evaluation.e3_v2.holdout import build
from aftersalesagent.evaluation.e3_v2.review import save
from aftersalesagent.local_config import load_local_config

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    offline = sub.add_parser("offline-holdout")
    offline.add_argument(
        "--directory", type=Path, default=ROOT / ".local/e3_v2_holdout_runs/offline_v1"
    )
    for name in ("preview", "freeze", "smoke", "remaining"):
        command = sub.add_parser(name)
        command.add_argument("--env-file", type=Path, required=True)
        command.add_argument(
            "--directory", type=Path, default=ROOT / ".local/e3_v2_live_runs/deepseek_20261027_v1"
        )
        if name == "freeze":
            command.add_argument("--output", type=Path, required=True)
        if name in {"smoke", "remaining"}:
            command.add_argument("--freeze-report", type=Path, required=True)
            command.add_argument("--enable-paid-api", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        manifest = build(ROOT)
        print(json.dumps({"suite": manifest["suite_id"], "tasks": 24, "real_api_calls": 0}))
        return
    if args.command == "offline-holdout":
        asyncio.run(offline_holdout(ROOT, args.directory))
        return
    settings, secret = load_local_config(args.env_file)
    if settings.backend != "deepseek" or not secret.get_secret_value():
        parser.error("Configure DeepSeek locally; never send the key in chat")
    config = freeze_config(ROOT, settings)
    print(
        json.dumps(
            {
                "command": args.command,
                "model": config["options"]["model"],
                "live_split": "dev",
                "episodes": 48,
                "maximum_model_requests": 576,
                "holdout_live_episodes": 0,
                "preview_api_calls": 0,
            }
        ),
        flush=True,
    )
    if args.command == "preview":
        return
    if args.command == "freeze":
        if args.output.exists():
            parser.error("Do not overwrite a freeze")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save(args.output, config)
        return
    if not args.enable_paid_api:
        parser.error("Explicit --enable-paid-api is required")
    if json.loads(args.freeze_report.read_text("utf-8")) != config:
        parser.error("Frozen model, sources, protocol, tasks, policy, Skill or budget changed")
    asyncio.run(
        live(
            ROOT,
            directory=args.directory,
            phase=args.command,
            settings=settings,
            secret=secret,
            config=config,
        )
    )


if __name__ == "__main__":
    main()
