"""固定 Skill 的本机人工管理；不读取 .env，不发模型请求。"""

import argparse
import asyncio
import json
from pathlib import Path

from aftersalesagent.skills.registry import SkillRegistry

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=ROOT / ".local/skills/registry.sqlite")
    sub = parser.add_subparsers(dest="command", required=True)
    install = sub.add_parser("install")
    install.add_argument("--bundle-root", type=Path, default=ROOT / "skills")
    install.add_argument("--skill-id", default="after-sales-workflow")
    install.add_argument("--version", default="1.0.0")
    sub.add_parser("list")
    sub.add_parser("history")
    show = sub.add_parser("show")
    show.add_argument("--skill-id", default="after-sales-workflow")
    show.add_argument("--version", default="released")
    activate = sub.add_parser("activate")
    activate.add_argument("--skill-id", default="after-sales-workflow")
    activate.add_argument("--version", required=True)
    activate.add_argument("--expected-current", required=True)
    activate.add_argument("--reason", required=True)
    args = parser.parse_args()
    registry = SkillRegistry(args.registry)
    if args.command == "install":
        result = registry.install(args.bundle_root, args.skill_id, args.version).metadata()
        output = result.model_dump(mode="json")
    elif args.command == "activate":
        output = registry.activate(
            args.skill_id,
            args.version,
            expected_current=args.expected_current,
            reason=args.reason,
        )
    elif args.command == "show":
        output = asyncio.run(registry.load(args.skill_id, args.version)).model_dump(mode="json")
    elif args.command == "list":
        output = [m.model_dump(mode="json") for m in asyncio.run(registry.list_metadata())]
    else:
        output = registry.history()
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
