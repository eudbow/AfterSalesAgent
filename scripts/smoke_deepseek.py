"""默认只检查配置；显式--live和--enable-paid-api才进行最多两次真实模型调用。"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import uuid
from pathlib import Path

from aftersalesagent.business.service import safe_calculate
from aftersalesagent.local_config import load_local_config, options_from_settings
from aftersalesagent.models.deepseek import DeepSeekBackend
from aftersalesagent.models.errors import ModelBackendFault
from aftersalesagent.schemas.contracts import ModelMessage, ModelRequest, ToolSpec

ROOT = Path(__file__).resolve().parents[1]


def fingerprint(options) -> str:
    return hashlib.sha256(options.model_dump_json().encode()).hexdigest()


async def live(options, secret) -> dict:
    responses = []
    backend = DeepSeekBackend(options, secret, allow_network=True)
    try:
        tool = ToolSpec(
            name="calculate",
            description="Calculate a safe arithmetic expression.",
            arguments_schema={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
                "additionalProperties": False,
            },
            effect="read",
        )
        messages = [
            ModelMessage(
                role="user",
                content=(
                    "Use the calculate tool once for 17 + 25. Then state the returned result."
                ),
            )
        ]
        first = await backend.generate(
            ModelRequest(
                messages=messages,
                tools=[tool],
                model_name=options.model,
                tool_choice="required",
                max_output_tokens=options.max_output_tokens,
                temperature=options.temperature,
            )
        )
        responses.append(first)
        if len(first.tool_calls) != 1 or first.tool_calls[0].name != "calculate":
            raise ValueError("Smoke test expected one calculate call")
        call = first.tool_calls[0]
        if set(call.arguments) != {"expression"} or not isinstance(
            call.arguments["expression"], str
        ):
            raise ValueError("Smoke test received unexpected arguments")
        answer = safe_calculate(call.arguments["expression"])
        if answer != "42.00":
            raise ValueError("Smoke test expression did not match requested calculation")
        messages.extend(
            [
                ModelMessage(role="assistant", content=first.content, tool_calls=first.tool_calls),
                ModelMessage(
                    role="tool", tool_call_id=call.call_id, content=json.dumps({"result": answer})
                ),
            ]
        )
        second = await backend.generate(
            ModelRequest(
                messages=messages,
                tools=[tool],
                model_name=options.model,
                tool_choice="none",
                max_output_tokens=options.max_output_tokens,
                temperature=options.temperature,
            )
        )
        responses.append(second)
        passed = (
            not second.tool_calls
            and "42" in second.content
            and all(r.input_tokens is not None and r.output_tokens is not None for r in responses)
        )
        return {
            "passed": passed,
            "responses": [r.model_dump(mode="json") for r in responses],
            "model_calls": 2,
            "usage_recorded": all(
                r.input_tokens is not None and r.output_tokens is not None for r in responses
            ),
        }
    except ModelBackendFault as exc:
        return {
            "passed": False,
            "error": exc.public(),
            "responses": [r.model_dump(mode="json") for r in responses],
            "model_calls": len(responses) + 1,
            "usage_recorded": False,
        }
    except ValueError:
        return {
            "passed": False,
            "error": {"code": "smoke_protocol_check_failed"},
            "responses": [r.model_dump(mode="json") for r in responses],
            "model_calls": len(responses),
            "usage_recorded": False,
        }
    finally:
        await backend.aclose()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--enable-paid-api", action="store_true")
    args = parser.parse_args()
    try:
        settings, secret = load_local_config(args.env_file)
        options = options_from_settings(settings)
    except (ValueError, OSError):
        raise SystemExit("Configuration incomplete/invalid. See docs/DEEPSEEK_SETUP.md.") from None
    if not args.live or args.check_config:
        print(
            json.dumps(
                {
                    "mode": "local_configuration_check",
                    "network_requests": 0,
                    "model": options.model,
                    "thinking": "disabled",
                    "key_configured": bool(secret.get_secret_value()),
                    "key_printed": False,
                    "settings_fingerprint": fingerprint(options),
                },
                indent=2,
            )
        )
        return
    if not args.enable_paid_api:
        parser.error("--live requires --enable-paid-api")
    if not secret.get_secret_value():
        parser.error("Fill DEEPSEEK_API_KEY locally; do not send it in chat")
    report = asyncio.run(live(options, secret))
    report.update(
        mode="live_tool_smoke",
        model=options.model,
        settings_fingerprint=fingerprint(options),
        options=options.model_dump(),
        cost_estimate=None,
    )
    target = ROOT / ".local/results" / ("deepseek_smoke_" + uuid.uuid4().hex[:10] + ".json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "model_calls": report["model_calls"],
                "result": str(target),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
