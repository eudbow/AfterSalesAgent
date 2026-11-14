"""Rebuild reviewed tool/policy contracts through static parsing; no upstream code execution."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from aftersalesagent.business.source_audit import verify_source

ROOT = Path(__file__).resolve().parents[1]
TOOL_NAMES = {
    "find_user_id_by_email",
    "find_user_id_by_name_zip",
    "get_user_details",
    "get_order_details",
    "get_product_details",
    "calculate",
    "cancel_pending_order",
    "modify_pending_order_address",
    "return_delivered_order_items",
    "transfer_to_human_agents",
}
WRITE_NAMES = {
    "cancel_pending_order",
    "modify_pending_order_address",
    "return_delivered_order_items",
    "transfer_to_human_agents",
}
RULES = [
    (
        "R01",
        "身份定位",
        "At the beginning of the conversation",
        "通过email或姓名加邮编定位；已给user_id仍需定位。",
    ),
    ("R02", "单用户会话", "You can only help one user", "只处理已定位用户的订单。"),
    ("R03", "明确确认", "Before taking any action", "写操作前展示具体细节并获得确认。"),
    (
        "R04",
        "取消订单",
        "## Cancel pending order",
        "仅pending；原因限no longer needed或ordered by mistake；退款原支付方式。",
    ),
    (
        "R05",
        "修改订单地址",
        "## Modify pending order",
        "仅pending订单；用户默认地址与订单地址是不同对象，MVP只改订单地址。",
    ),
    (
        "R06",
        "提交退货",
        "## Return delivered order",
        "仅delivered；收集完整item列表；退款原方式或已有gift_card；结果return requested。",
    ),
    (
        "R07",
        "人工升级",
        "To transfer, first make a tool call",
        "能力范围外调用转人工工具并给出状态；后续只模拟升级记录。",
    ),
    (
        "R08",
        "一次性商品变更",
        "Exchange or modify order tools",
        "退货/换货与商品修改应按原工具具体状态限制处理，不能自行猜测复用条件。",
    ),
]


def write_json(name: str, value: object) -> None:
    (ROOT / "data/manifests" / name).write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    errors = verify_source(ROOT)
    if errors:
        raise SystemExit("\n".join(errors))
    source = ROOT / "data/upstream/tau2-bench/src/tau2/domains/retail/tools.py"
    methods = {
        node.name: node
        for node in ast.walk(ast.parse(source.read_text("utf-8")))
        if isinstance(node, ast.FunctionDef)
    }
    rows = []
    for name in sorted(TOOL_NAMES):
        node = methods[name]
        properties = {}
        for argument in node.args.args:
            if argument.arg == "self":
                continue
            if argument.annotation is None:
                raise ValueError(f"Unannotated argument: {name}/{argument.arg}")
            annotation = ast.unparse(argument.annotation)
            if annotation == "str":
                properties[argument.arg] = {"type": "string"}
            elif annotation in ("List[str]", "list[str]"):
                properties[argument.arg] = {"type": "array", "items": {"type": "string"}}
            else:
                raise ValueError(f"Unhandled argument type: {annotation}")
        rows.append(
            {
                "spec": {
                    "name": name,
                    "description": (ast.get_docstring(node) or "").split("Args:")[0].strip(),
                    "arguments_schema": {
                        "type": "object",
                        "properties": properties,
                        "required": list(properties),
                        "additionalProperties": False,
                    },
                    "effect": "write" if name in WRITE_NAMES else "read",
                },
                "upstream_source": source.relative_to(ROOT).as_posix(),
                "source_lines": [node.lineno, node.end_lineno],
                "server_context_not_model_arguments": [
                    "authenticated_user_id",
                    "environment_id",
                    "approved_action_id",
                    "idempotency_key",
                ],
                "implementation_stage": "M1/M2; not executable in M0",
            }
        )
    write_json("tool_contracts.json", rows)
    policy = ROOT / "data/upstream/tau2-bench/data/tau2/domains/retail/policy.md"
    lines = policy.read_text("utf-8").splitlines()
    reviewed = []
    for rid, title, needle, meaning in RULES:
        matches = [index + 1 for index, line in enumerate(lines) if needle in line]
        if not matches:
            raise ValueError(f"Missing policy anchor: {rid}")
        reviewed.append(
            {
                "rule_id": rid,
                "title": title,
                "source_path": policy.relative_to(ROOT).as_posix(),
                "source_line": matches[0],
                "interpretation": meaning,
                "source_sha256": hashlib.sha256(policy.read_bytes()).hexdigest(),
            }
        )
    write_json("policy_rules.json", reviewed)
    print(f"Prepared {len(rows)} tool contracts and {len(reviewed)} policy review records")


if __name__ == "__main__":
    main()
