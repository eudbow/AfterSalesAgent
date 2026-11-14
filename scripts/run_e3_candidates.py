"""有限v2.1诊断、开发轨迹投影与隔离候选；默认不联网。"""

import argparse
import asyncio
import json
from pathlib import Path

from aftersalesagent.evaluation.e3_candidates.evidence import (
    checked_packet,
    export_evidence,
    review_packet,
)
from aftersalesagent.evaluation.e3_candidates.generation import (
    OfflineGenerator,
    generate,
    request_for,
)
from aftersalesagent.evaluation.e3_candidates.workflow import (
    diagnose,
    frozen_config,
    offline,
)
from aftersalesagent.evaluation.e3_v2.review import save
from aftersalesagent.local_config import load_local_config, options_from_settings
from aftersalesagent.models.deepseek import DeepSeekBackend

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("export")
    ex.add_argument("--trace", action="append", required=True)
    ex.add_argument("--output", type=Path, required=True)
    review = sub.add_parser("approve-evidence")
    review.add_argument("--evidence", type=Path, required=True)
    review.add_argument("--packet-sha256", required=True)
    review.add_argument("--reviewer", required=True)
    off = sub.add_parser("offline")
    off.add_argument("--directory", type=Path, required=True)
    for command in ("generator-offline", "preview", "freeze", "smoke", "diagnose", "generate"):
        p = sub.add_parser(command)
        p.add_argument("--evidence", type=Path, required=True)
        p.add_argument("--directory", type=Path, required=True)
        if command != "generator-offline":
            p.add_argument("--env-file", type=Path, required=True)
        if command == "freeze":
            p.add_argument("--output", type=Path, required=True)
        if command in {"smoke", "diagnose", "generate"}:
            p.add_argument("--freeze-report", type=Path, required=True)
            p.add_argument("--enable-paid-api", action="store_true")
        if command == "generate":
            p.add_argument("--version", choices=("1.0.1", "1.0.2"), default="1.0.1")
    args = parser.parse_args()
    if args.command == "export":
        print(
            export_evidence(
                ROOT, ROOT / "configs/baselines/e3_v2_s0_s1_dev_v1.json", args.trace, args.output
            )
        )
        return
    if args.command == "approve-evidence":
        print(
            review_packet(args.evidence, packet_sha256=args.packet_sha256, reviewer=args.reviewer)
        )
        return
    if args.command == "offline":
        print(asyncio.run(offline(ROOT, args.directory)))
        return
    if args.command == "generator-offline":
        request = request_for(ROOT, checked_packet(args.evidence), model="offline-fixture")
        print(
            asyncio.run(
                generate(
                    ROOT,
                    evidence=args.evidence,
                    directory=args.directory,
                    backend=OfflineGenerator(),
                    request=request,
                )
            )
        )
        return
    settings, secret = load_local_config(args.env_file)
    if settings.backend != "deepseek" or not secret.get_secret_value():
        parser.error("Configure DeepSeek locally; do not print keys")
    config = frozen_config(ROOT, settings, args.evidence)
    if args.command == "preview":
        print(
            json.dumps(
                {
                    "episodes": 12,
                    "maximum_requests": 146,
                    "model": settings.model_name,
                    "actual_api_calls": 0,
                    "holdout_calls": 0,
                }
            )
        )
        return
    if args.command == "freeze":
        if args.output.exists():
            parser.error("Never overwrite a freeze")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save(args.output, config)
        return
    if not args.enable_paid_api:
        parser.error("Explicit --enable-paid-api required")
    if json.loads(args.freeze_report.read_text("utf-8")) != config:
        parser.error("Source/model/protocol/evidence changed after freeze")
    if args.command in {"smoke", "diagnose"}:
        asyncio.run(
            diagnose(
                ROOT,
                directory=args.directory,
                config=config,
                settings=settings,
                secret=secret,
                phase=args.command,
            )
        )
        return
    if not (args.directory / "diagnostics/completed.json").exists():
        parser.error("Finish the planned diagnostics before candidate generation")
    backend = DeepSeekBackend(options_from_settings(settings), secret, allow_network=True)

    async def run_generation():
        try:
            return await generate(
                ROOT,
                evidence=args.evidence,
                directory=args.directory / "generation" / args.version,
                backend=backend,
                request=request_for(ROOT, checked_packet(args.evidence), model=settings.model_name),
                version=args.version,
            )
        finally:
            await backend.aclose()

    print(json.dumps(asyncio.run(run_generation()), ensure_ascii=False))


if __name__ == "__main__":
    main()
