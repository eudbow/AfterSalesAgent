"""冻结门槛下的S1/S2对照；默认离线，真实API要求显式冻结与开关。"""

import argparse
import asyncio
import json
import os
from pathlib import Path

from aftersalesagent.evaluation.e3_candidates.evidence import safe_local
from aftersalesagent.evaluation.e3_comparison.execution import freeze, live, offline, save_decision
from aftersalesagent.evaluation.e3_comparison.materials import build
from aftersalesagent.evaluation.e3_comparison.review import import_review
from aftersalesagent.evaluation.e3_v2.review import save
from aftersalesagent.local_config import load_local_config

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    off = sub.add_parser("offline")
    off.add_argument("--directory", type=Path, required=True)
    review = sub.add_parser("import-review")
    review.add_argument("--stage-directory", type=Path, required=True)
    review.add_argument("--sheet", type=Path, required=True)
    review.add_argument("--reviewer", required=True)
    review.add_argument(
        "--reviewer-kind", choices=("assistant", "human", "test_fixture"), required=True
    )
    decision = sub.add_parser("decide")
    decision.add_argument("--directory", type=Path, required=True)
    decision.add_argument("--split", choices=("development", "validation", "test"), required=True)
    for name in ("preview", "freeze", "smoke", "development", "validation", "test"):
        command = sub.add_parser(name)
        command.add_argument("--env-file", type=Path, required=True)
        command.add_argument("--directory", type=Path, required=True)
        if name == "freeze":
            command.add_argument("--output", type=Path, required=True)
        elif name != "preview":
            command.add_argument("--freeze-report", type=Path, required=True)
            command.add_argument("--enable-paid-api", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        manifest = build(ROOT)
        print(json.dumps({"suite": manifest["suite_id"], "tasks": 42, "actual_api_calls": 0}))
        return
    if args.command == "offline":
        print(asyncio.run(offline(ROOT, args.directory)))
        return
    if args.command == "import-review":
        print(
            import_review(
                args.stage_directory,
                reviewed_sheet=args.sheet,
                reviewer=args.reviewer,
                reviewer_kind=args.reviewer_kind,
            )
        )
        return
    if args.command == "decide":
        stage = args.directory / args.split
        if not (stage / "halt.json").exists() and not (stage / "review/adjudication.json").exists():
            parser.error("Finish explicit review before committing a stage decision")
        verdict = save_decision(ROOT, args.directory, args.split)
        print(json.dumps(verdict, ensure_ascii=False))
        return
    settings, secret = load_local_config(args.env_file)
    if not secret.get_secret_value():
        parser.error("Configure API key locally; never print it")
    config = freeze(ROOT, settings)
    if args.command == "preview":
        print(
            json.dumps(
                {
                    "development_episodes": 72,
                    "maximum_all_stage_episodes": 168,
                    "actual_api_calls": 0,
                    "automatic_release": False,
                }
            )
        )
        return
    if args.command == "freeze":
        if args.output.exists():
            parser.error("Never overwrite experiment freeze")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save(args.output, config)
        return
    if not args.enable_paid_api:
        parser.error("Explicit --enable-paid-api required")
    if json.loads(args.freeze_report.read_text("utf-8")) != config:
        parser.error("Sources/model/materials/Skill/rules changed after freeze")
    split = "development" if args.command == "smoke" else args.command
    args.directory = safe_local(ROOT, args.directory)
    args.directory.mkdir(parents=True, exist_ok=True)
    lock = args.directory / "run.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(descriptor)
    try:
        print(
            json.dumps(
                asyncio.run(
                    live(
                        ROOT,
                        args.directory,
                        split,
                        "smoke" if args.command == "smoke" else "remaining",
                        settings,
                        secret,
                        config,
                    )
                ),
                ensure_ascii=False,
            )
        )
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
