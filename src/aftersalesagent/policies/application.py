"""网页与固定评测共用同一政策及应用工具协议，原始政策文件保持不变。"""

from pathlib import Path

# 上游工具会直接写订单；本应用同名工具只准备提议，必须明确说明这个适配差异。
APPLICATION_PROTOCOL = (
    "\nApplication tool adaptation (applies to the policy above): use one tool at a time. "
    "The functions cancel_pending_order, modify_pending_order_address, and "
    "return_delivered_order_items ONLY PREPARE a proposed action; calling them does NOT "
    "modify the order, issue a refund, or count as customer approval. "
    "After authenticating the customer, reading the order, and collecting valid arguments, "
    "call the matching prepare function immediately to create the application's confirmation "
    "card. Do not wait for a conversational yes before preparing that card. "
    "The application shows the exact proposal and requires a separate explicit confirmation "
    "event before any business change; the model cannot approve or bypass that event. "
    "A text-only question asking for confirmation cannot create a confirmation card. "
    "Do not claim a prepared action has already executed. "
    "Distinguish a return request from a completed refund."
)


def load_application_policy(root: Path) -> str:
    raw = (root / "data/upstream/tau2-bench/data/tau2/domains/retail/policy.md").read_text(
        encoding="utf-8"
    )
    return raw + APPLICATION_PROTOCOL
