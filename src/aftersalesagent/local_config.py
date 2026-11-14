"""仅显式指定时读取本地.env；不执行插值/命令，不打印密钥或文件内容。"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import SecretStr

from aftersalesagent.config import Settings
from aftersalesagent.models.deepseek import DeepSeekOptions


def load_local_config(path: Path | None = None) -> tuple[Settings, SecretStr]:
    values = dict(os.environ)
    if path is not None:
        for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            if "=" not in text:
                raise ValueError(f"Invalid local config syntax at line {number}")
            key, value = text.split("=", 1)
            key, value = key.strip(), value.strip()
            if not (key.startswith("AFTERSALES_") or key == "DEEPSEEK_API_KEY"):
                raise ValueError(f"Unsupported local config field at line {number}")
            if value.startswith(("'", '"')):
                if len(value) < 2 or value[-1] != value[0]:
                    raise ValueError(f"Unclosed local config quote at line {number}")
                value = value[1:-1]
            # 显式文件覆盖同名进程配置；不修改os.environ。
            values[key] = value
    settings = Settings.from_environment(values)
    return settings, SecretStr(values.get("DEEPSEEK_API_KEY", "").strip())


def options_from_settings(settings: Settings) -> DeepSeekOptions:
    if not settings.model_name:
        raise ValueError("Set AFTERSALES_MODEL_NAME explicitly using current DeepSeek docs")
    return DeepSeekOptions(
        model=settings.model_name,
        base_url=settings.model_base_url,
        timeout_seconds=settings.model_timeout_seconds,
        max_output_tokens=settings.model_max_output_tokens,
        temperature=settings.model_temperature,
    )
