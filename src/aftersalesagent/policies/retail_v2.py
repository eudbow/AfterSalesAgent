"""Versioned application facts; policy retrieval cannot expand these capabilities."""

import hashlib
import json
import re

CAPABILITIES = {
    "id": "APP-CAP-V2",
    "version": "2.0.0",
    "source": "application capability mapping to RetailService and ToolRegistry",
    "prepare_tools": [
        "cancel_pending_order",
        "modify_pending_order_address",
        "return_delivered_order_items",
    ],
    "other_write_tools": ["transfer_to_human_agents"],
    "unsupported": [
        "exchange",
        "change_payment",
        "modify_items",
        "update_default_address",
        "reactivate_order",
        "send_email",
    ],
    "result_semantics": {
        "prepare": "只生成确认卡，尚未执行；独立确认后后端重新校验并执行。",
        "return requested": "退货申请已提交，不代表外部退款到账。",
        "refund_record": "模拟支付历史中的退款记录，不能证明外部支付机构到账。",
        "processed": "已处理；仅凭此状态不能判断是否发货。",
        "handoff_requested": "模拟交接请求已记录，不代表真人已经接通。",
    },
}


def capability_hash():
    return hashlib.sha256(
        json.dumps(CAPABILITIES, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def validate_tools(specs):
    actual = {spec.name for spec in specs if spec.effect == "write"}
    expected = set(CAPABILITIES["prepare_tools"] + CAPABILITIES["other_write_tools"])
    if actual != expected:
        raise ValueError(f"Retail capability/tool mismatch: {sorted(actual ^ expected)}")


def capability_prompt():
    return (
        "\n服务端固定提供的应用能力 [cite:APP-CAP-V2]，版本 2.0.0：\n"
        + json.dumps(CAPABILITIES, ensure_ascii=False)
        + "\n上游商家政策允许某项操作，不表示当前应用已实现。能力说明优先决定本应用能否办理，"
        "不得改写上游商家资格。中文用户用中文回答；政策搜索优先使用中文问题加动作/状态别名。"
        "不要披露无关的完整地址、支付标识或物流信息。"
        "资料齐备即准备确认卡，不要先要求口头确认才生成卡片。"
        "模拟人工交接不能承诺真人跟进或实际解决问题。"
        "取消原因必须来自用户原话，不能把赔偿等无效原因改写成不再需要或误下单。"
        "应用结果含义（包括模拟退款记录不能证明到账）请引用 APP-CAP-V2。"
        "取消资格必须检索并引用取消条款，能力说明不能代替商家条款。"
        "不要自行建议联系承运方、拒收等资料没有提供的流程。"
        "不要承诺人工跟进或由真人办理，模拟交接仅记录请求。"
    )


def cancellation_reason_supported(reason, user_messages):
    patterns = {
        "no longer needed": r"不再需要|不需要了|不要了|不想要|no longer need|don't need",
        "ordered by mistake": r"下错|误下|误买|买错|ordered by mistake|accidentally ordered",
    }
    return reason in patterns and bool(re.search(patterns[reason], user_messages, re.I))


def answer_issue(content):
    """Narrow checks for observed failures, not a universal entailment verifier."""
    ids = set(re.findall(r"\[cite:([^\]]+)\]", content))
    if (
        "取消" in content
        and re.search(r"(?:只|仅).{0,35}pending|pending.{0,20}(?:才能|才可)", content, re.I)
        and not ids.intersection({"POL-07", "03_cancellation", "POL-01"})
    ):
        return "取消资格必须引用实际检索的取消政策，不能仅引用应用能力。"
    if re.search(r"(?:建议|可以|请).{0,10}联系承运|拒收", content):
        return "删除资料未支持的联系承运方或拒收建议。"
    if re.search(r"(?:人工|同事).{0,18}(?:跟进|处理此事|办理换货)", content):
        if "模拟" not in content:
            return "人工交接仅记录模拟请求，不能承诺实际跟进或办理。"
    for sentence in re.split(r"[。！？\n]", content):
        if re.search(r"退款.{0,15}(?:已完成|已到账)|已全额退[回款]", sentence):
            if not re.search(r"不|无法|未能|并非|不能", sentence):
                return "模拟退款记录不能证明外部到账或实际退款完成。"
    return None
