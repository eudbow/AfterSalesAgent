"""启动本地单进程服务；显式配置与开关才允许付费模型调用。"""

import argparse
from pathlib import Path

import uvicorn

from aftersalesagent.api.app import create_app
from aftersalesagent.local_config import load_local_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--enable-paid-api", action="store_true")
    parser.add_argument("--tool-transport", choices=["native", "mcp"])
    parser.add_argument("--knowledge-transport", choices=["native", "mcp"])
    parser.add_argument("--knowledge-config", type=Path)
    parser.add_argument("--skill-id")
    parser.add_argument("--skill-version")
    args = parser.parse_args()
    settings, secret = load_local_config(args.env_file)
    overrides = {
        k: v
        for k, v in {
            "tool_transport": args.tool_transport,
            "knowledge_transport": args.knowledge_transport,
            "knowledge_config": str(args.knowledge_config) if args.knowledge_config else None,
            "skill_id": args.skill_id,
            "skill_version": args.skill_version,
        }.items()
        if v is not None
    }
    settings = type(settings).model_validate(settings.model_dump() | overrides)
    app = create_app(
        data_dir=args.data_dir,
        settings=settings,
        api_key=secret,
        enable_paid_api=args.enable_paid_api,
    )
    uvicorn.run(app, host="127.0.0.1", port=args.port, workers=1)
