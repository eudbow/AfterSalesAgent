"""正式协议v2工程入口：只支持预览/离线回放，不加载Key或执行真实API。"""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from aftersalesagent.business.database import import_pinned_database
from aftersalesagent.evaluation.e3_v2.contracts import SimulatorOptions
from aftersalesagent.evaluation.e3_v2.replay import ProtocolReplay
from aftersalesagent.evaluation.e3_v2.review import export_review, import_review
from aftersalesagent.evaluation.e3_v2.runner import run_case
from aftersalesagent.evaluation.e3_v2.scoring import aggregate
from aftersalesagent.evaluation.e3_v2.suite import build, load, load_protocol
from aftersalesagent.evaluation.e3_v2.user import FormalUser
from aftersalesagent.policies.application import load_application_policy
from aftersalesagent.schemas.contracts import RunContext, RunLimits
from aftersalesagent.skills.registry import SkillRegistry

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


async def replay(args):
    cases, manifest = load(ROOT)
    protocol = load_protocol(ROOT)
    output = args.directory.resolve()
    if not output.is_relative_to((ROOT / ".local").resolve()) or output.exists():
        raise ValueError("Choose a new output directory under project .local")
    output.mkdir(parents=True)
    files = [
        *ROOT.joinpath("src").rglob("*.py"),
        Path(__file__).resolve(),
        ROOT / manifest["protocol_file"],
        ROOT / "configs/m3_live_limits.json",
        ROOT / "skills/after-sales-workflow/1.0.0/SKILL.md",
        ROOT / "skills/after-sales-workflow/1.0.0/manifest.json",
    ]
    save(
        output / "config.json",
        {
            "protocol": protocol,
            "suite": manifest,
            "backend": "ProtocolReplay",
            "variant": args.variant,
            "transport": args.transport,
            "actual_api_calls": 0,
            "sources": {
                p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in files
            },
        },
    )
    template = ROOT / ".local/databases/retail_template.sqlite"
    if not template.exists():
        import_pinned_database(ROOT, template).close()
    options = SimulatorOptions.model_validate(
        {
            key: protocol["simulator"][key]
            for key in ("per_slot_per_stage", "per_stage", "total_optional_replies")
        }
    )
    skill = SkillRegistry(output / "skills.sqlite")
    skill.install(ROOT / "skills", "after-sales-workflow", "1.0.0")
    rows = []
    for case in cases:
        for group in ("S0", "S1"):
            row = await run_case(
                case,
                folder=output / f"{case['public'].task_id}_{group}",
                template=template,
                backend_factory=lambda case=case: ProtocolReplay(
                    case["script"], variant=args.variant
                ),
                user_factory=lambda script: FormalUser(script, options=options),
                skill_provider=skill if group == "S1" else None,
                policy_text=load_application_policy(ROOT),
                context=RunContext(
                    environment_id="assigned",
                    limits=RunLimits.model_validate_json(
                        (ROOT / "configs/m3_live_limits.json").read_bytes()
                    ),
                    skill_id="after-sales-workflow" if group == "S1" else None,
                    skill_version="1.0.0" if group == "S1" else "none",
                ),
                group=group,
                transport=args.transport,
                protocol=protocol,
            )
            rows.append(row)
            save(
                output / "summary.json",
                {
                    "backend": "ProtocolReplay",
                    "variant": args.variant,
                    "episodes": len(rows),
                    "actual_api_calls": 0,
                    "local_gpu_experiments": 0,
                    "meaning": (
                        "Engineering replay, not language or business ability of a real model"
                    ),
                    "groups": aggregate(rows),
                },
            )
            print(
                json.dumps(
                    {
                        "episode": f"{case['public'].task_id}_{group}",
                        "status": row["scoring"]["task_status"],
                        "optional_replies": row["scoring"]["interaction"]["optional_replies"],
                    }
                ),
                flush=True,
            )
    export_review(output, output / "language_review", protocol)
    print("Saved:", output / "summary.json", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    sub.add_parser("preview")
    run = sub.add_parser("replay")
    run.add_argument("--directory", type=Path, default=ROOT / ".local/e3_v2_runs/replay_v1")
    run.add_argument(
        "--variant",
        choices=["direct", "clarify", "wrong_language", "coverage_gap", "repeat"],
        default="clarify",
    )
    run.add_argument("--transport", choices=["native", "mcp"], default="mcp")
    export = sub.add_parser("export-review")
    export.add_argument("--run-directory", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    review = sub.add_parser("import-review")
    review.add_argument("--sheet", type=Path, required=True)
    review.add_argument("--private-map", type=Path, required=True)
    review.add_argument("--output", type=Path, required=True)
    review.add_argument("--reviewer", required=True)
    review.add_argument(
        "--reviewer-kind", choices=["human", "assistant", "test_fixture"], required=True
    )
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(build(ROOT), ensure_ascii=False, indent=2))
    elif args.command == "preview":
        cases, manifest = load(ROOT)
        print(
            json.dumps(
                {
                    "protocol": manifest["protocol_id"],
                    "tasks": len(cases),
                    "language_groups": manifest["language_groups"],
                    "actual_api_calls": 0,
                }
            )
        )
    elif args.command == "replay":
        asyncio.run(replay(args))
    elif args.command == "export-review":
        print(json.dumps(export_review(args.run_directory, args.output, load_protocol(ROOT))))
    else:
        result = import_review(
            args.sheet,
            args.private_map,
            args.output,
            reviewer=args.reviewer,
            reviewer_kind=args.reviewer_kind,
            protocol=load_protocol(ROOT),
        )
        print(json.dumps({"status": result["status"], "groups": result["groups"]}))


if __name__ == "__main__":
    main()
