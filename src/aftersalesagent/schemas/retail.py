"""源数据的本地类型映射；只保留业务结构，不执行上游 Python 文件。"""

from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import Field

from aftersalesagent.schemas.contracts import Contract


class Address(Contract):
    address1: str
    address2: str
    city: str
    country: str
    state: str
    zip: str


class UserName(Contract):
    first_name: str
    last_name: str


class PaymentMethod(Contract):
    source: Literal["credit_card", "paypal", "gift_card"]
    id: str
    brand: str | None = None
    last_four: str | None = None
    balance: Decimal | None = None


class RetailUser(Contract):
    user_id: str
    name: UserName
    address: Address
    email: str
    payment_methods: dict[str, PaymentMethod]
    orders: list[str]


class Variant(Contract):
    item_id: str
    options: dict[str, str]
    available: bool
    price: Decimal


class Product(Contract):
    name: str
    product_id: str
    variants: dict[str, Variant]


class OrderItem(Contract):
    # item_id 不是订单行唯一键；M1 用 order_id + line_index 保存重复商品。
    name: str
    product_id: str
    item_id: str
    price: Decimal
    options: dict[str, str]


class Fulfillment(Contract):
    tracking_id: list[str]
    item_ids: list[str]


class PaymentRecord(Contract):
    transaction_type: Literal["payment", "refund"]
    amount: Decimal
    payment_method_id: str


class RetailOrder(Contract):
    order_id: str
    user_id: str
    address: Address
    items: list[OrderItem]
    status: Literal[
        "processed",
        "pending",
        "pending (item modified)",
        "delivered",
        "cancelled",
        "exchange requested",
        "return requested",
    ]
    fulfillments: list[Fulfillment]
    payment_history: list[PaymentRecord]
    cancel_reason: Literal["no longer needed", "ordered by mistake"] | None = None
    exchange_items: list[str] | None = None
    exchange_new_items: list[str] | None = None
    exchange_payment_method_id: str | None = None
    exchange_price_difference: Decimal | None = None
    return_items: list[str] | None = None
    return_payment_method_id: str | None = None


class RetailSnapshot(Contract):
    products: dict[str, Product]
    users: dict[str, RetailUser]
    orders: dict[str, RetailOrder]


class SampleReview(Contract):
    sample_id: str
    source_task_id: str
    source_split: Literal["train"] = "train"
    development_only: Literal[True] = True
    business_tags: list[str]
    family_id: str
    user_id: str
    relevant_order_ids: list[str]
    review_status: Literal["reviewed"] = "reviewed"
    review_notes: list[str]
    # 私有初始化/用户模拟器/判分材料另存，不进入 VisibleTaskInput。
    private_fixture: str
    policy_rule_ids: list[str]
    source_commit: str = Field(pattern=r"^[a-f0-9]{40}$")
