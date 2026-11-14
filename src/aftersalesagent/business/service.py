"""Deterministic policy + confirmation gate. Only this layer changes business state."""

from __future__ import annotations

import ast
import hashlib
import json
import operator
import uuid
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Literal, NoReturn

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import func, insert, select, update
from sqlalchemy.engine import Connection

from aftersalesagent.business.database import (
    RetailStore,
    cents,
    handoffs,
    invocations,
    methods,
    money,
    operations,
    orders,
    payments,
    pending,
    products,
    sessions,
    users,
    variants,
)
from aftersalesagent.schemas.contracts import (
    PendingAction,
    RunContext,
    SessionView,
    ToolCall,
    ToolError,
    ToolResult,
    ToolSpec,
)
from aftersalesagent.schemas.retail import Address


class Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class EmailArgs(Arguments):
    email: str = Field(min_length=1)


class NameArgs(Arguments):
    first_name: str
    last_name: str
    zip: str


class UserArgs(Arguments):
    user_id: str


class OrderArgs(Arguments):
    # 订单号是不透明标识，#是值的一部分；说明对所有继承的订单工具一致生效。
    order_id: str = Field(
        description="Exact order identifier copied from the customer or an order lookup. "
        "Preserve the leading # and every character, for example #W1234567. "
        "Do not strip #, rewrite the identifier, or use a different order."
    )


class ProductArgs(Arguments):
    product_id: str


class CalculateArgs(Arguments):
    expression: str = Field(min_length=1, max_length=256)


class CancelArgs(OrderArgs):
    reason: Literal["no longer needed", "ordered by mistake"]


class AddressArgs(OrderArgs):
    address1: str = Field(min_length=1)
    address2: str
    city: str = Field(min_length=1)
    state: str = Field(min_length=1)
    country: str = Field(min_length=1)
    zip: str = Field(min_length=1)


class ReturnArgs(OrderArgs):
    item_ids: list[str] = Field(min_length=1)
    payment_method_id: str


class HandoffArgs(Arguments):
    summary: str = Field(min_length=1, max_length=2000)


ARGUMENTS: dict[str, type[Arguments]] = {
    "find_user_id_by_email": EmailArgs,
    "find_user_id_by_name_zip": NameArgs,
    "get_user_details": UserArgs,
    "get_order_details": OrderArgs,
    "get_product_details": ProductArgs,
    "calculate": CalculateArgs,
    "cancel_pending_order": CancelArgs,
    "modify_pending_order_address": AddressArgs,
    "return_delivered_order_items": ReturnArgs,
    "transfer_to_human_agents": HandoffArgs,
}
TOOL_DESCRIPTIONS = {
    "find_user_id_by_email": "Locate the customer by email before accessing orders.",
    "find_user_id_by_name_zip": "Locate the customer by first/last name and default-address zip.",
    "get_user_details": "Read the located customer's details, orders, and owned payment methods.",
    "get_order_details": "Read an order owned by the located customer, including state and items.",
    "get_product_details": "Read product variants/options, not order ownership.",
    "calculate": "Safely calculate a numeric expression using + - * / and parentheses.",
    "cancel_pending_order": "Prepare cancellation of a pending order using an allowed reason.",
    "modify_pending_order_address": "Prepare a pending-order address change; supply all fields.",
    "return_delivered_order_items": "Prepare a return with item IDs and a refund method.",
    "transfer_to_human_agents": "Request a simulated human handoff with a factual summary.",
}
ORDER_WRITES = {
    "cancel_pending_order",
    "modify_pending_order_address",
    "return_delivered_order_items",
}


class BusinessFault(Exception):
    def __init__(self, code: str, message: str):
        self.error = ToolError.model_validate({"code": code, "message": message})
        super().__init__(message)


def fail(code: str, message: str) -> NoReturn:
    raise BusinessFault(code, message)


def safe_calculate(expression: str) -> str:
    """只解释受限AST；拒绝函数、属性、幂、整除、过长/过大表达式。"""
    tree = ast.parse(expression, mode="eval")
    if len(list(ast.walk(tree))) > 64:
        raise ValueError("Expression too complex")
    binary = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }

    def evaluate(node: ast.AST) -> Decimal:
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            result = Decimal(str(node.value))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            result = evaluate(node.operand)
            if isinstance(node.op, ast.USub):
                result = -result
        elif isinstance(node, ast.BinOp) and type(node.op) in binary:
            result = binary[type(node.op)](evaluate(node.left), evaluate(node.right))
        else:
            raise ValueError("Only numeric + - * / and parentheses are allowed")
        if not result.is_finite() or abs(result) > Decimal("1e12"):
            raise ValueError("Numeric result out of range")
        return result

    return format(evaluate(tree.body).quantize(Decimal("0.01")), ".2f")


def summarize_action(name: str, args: dict) -> str:
    oid = args["order_id"]
    if name == "cancel_pending_order":
        reason = {"ordered by mistake": "下错单", "no longer needed": "不再需要"}[args["reason"]]
        return f"将取消订单 {oid}，原因：{reason}。请核对订单后确认。"
    if name == "modify_pending_order_address":
        address = " · ".join(
            args[key]
            for key in ("address1", "address2", "city", "state", "country", "zip")
            if args[key]
        )
        return f"将订单 {oid} 的收货地址改为：{address}。请核对新地址后确认。"
    return (
        f"将为订单 {oid} 提交 {len(args['item_ids'])} 件商品的退货申请，"
        f"退款方式：{args['payment_method_id']}。提交申请不代表退款到账。"
    )


class RetailService:
    def __init__(self, store: RetailStore, *, clock: Callable[[], datetime] | None = None):
        self.store = store
        self.environment_id = store.path.stem
        self.clock = clock or (lambda: datetime.now(UTC))

    def specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name=name,
                description=TOOL_DESCRIPTIONS[name]
                + (
                    "; call before asking for confirmation to create a confirmation card. "
                    "This only prepares the proposal and never changes the order. "
                    "The application separately requires explicit customer approval to commit"
                    if name in ORDER_WRITES
                    else ""
                ),
                arguments_schema=cls.model_json_schema(),
                effect="write"
                if name in ORDER_WRITES or name == "transfer_to_human_agents"
                else "read",
            )
            for name, cls in ARGUMENTS.items()
        ]

    def start(self, context: RunContext) -> SessionView:
        if context.environment_id != self.environment_id:
            raise ValueError("Runtime context must select this store environment")
        # M1模板可能早于本表；只增建幂等记录表，不重写既有业务数据。
        invocations.create(self.store.engine, checkfirst=True)
        sid = uuid.uuid4().hex
        with self.store.transaction() as connection:
            connection.execute(
                insert(sessions).values(
                    session_id=sid,
                    environment_id=context.environment_id,
                    context_json=context.model_dump(mode="json"),
                )
            )
        return self.view(sid)

    def _session(self, connection: Connection, session_id: str):
        row = (
            connection.execute(select(sessions).where(sessions.c.session_id == session_id))
            .mappings()
            .first()
        )
        if row is None:
            fail("not_found_or_forbidden", "Session not found")
        assert row is not None
        return row

    def _owner(self, connection: Connection, session_id: str) -> str:
        uid = self._session(connection, session_id)["authenticated_user_id"]
        if not uid:
            fail("policy_denied", "Locate the user by email or name and zip first")
        return uid

    def _owned_order(self, connection: Connection, session_id: str, order_id: str) -> dict:
        uid = self._owner(connection, session_id)
        order = self.store.order(connection, order_id)
        if order is None or order["user_id"] != uid:
            fail("not_found_or_forbidden", "Order not found or unavailable")
        assert order is not None
        return order

    def _pending(self, connection: Connection, session_id: str) -> PendingAction | None:
        row = connection.execute(
            select(pending.c.payload).where(
                pending.c.session_id == session_id, pending.c.status == "prepared"
            )
        ).first()
        if row is None:
            return None
        action = PendingAction.model_validate(row[0])
        return action if action.expires_at > self.clock() else None

    def view(self, session_id: str) -> SessionView:
        with self.store.transaction() as connection:
            row = self._session(connection, session_id)
            action = self._pending(connection, session_id)
            return SessionView(
                session_id=session_id,
                environment_id=row["environment_id"],
                authenticated_user_id=row["authenticated_user_id"],
                pending_action=action,
                status="waiting_confirmation" if action else "ready",
            )

    def visible_orders(self, session_id: str) -> list[dict]:
        with self.store.transaction() as connection:
            uid = self._session(connection, session_id)["authenticated_user_id"]
            if uid is None:
                return []
            result = [
                self.store.order(connection, oid)
                for oid in connection.scalars(
                    select(orders.c.order_id)
                    .where(orders.c.user_id == uid)
                    .order_by(orders.c.order_id)
                )
            ]
            return [order for order in result if order is not None]

    def invalidate_pending(self, session_id: str) -> None:
        with self.store.transaction() as connection:
            self._session(connection, session_id)
            connection.execute(
                update(pending)
                .where(pending.c.session_id == session_id, pending.c.status == "prepared")
                .values(status="superseded")
            )

    def dispatch(
        self, session_id: str, call: ToolCall, *, idempotency_key: str | None = None
    ) -> ToolResult:
        try:
            cls = ARGUMENTS.get(call.name)
            if cls is None:
                fail("invalid_arguments", "Unknown tool")
            assert cls is not None
            args = cls.model_validate(call.arguments).model_dump()
            with self.store.transaction(immediate=True) as connection:
                self._session(connection, session_id)
                fingerprint = hashlib.sha256(
                    json.dumps({"name": call.name, "arguments": args}, sort_keys=True).encode()
                ).hexdigest()
                if idempotency_key:
                    previous = (
                        connection.execute(
                            select(invocations).where(
                                invocations.c.session_id == session_id,
                                invocations.c.invocation_key == idempotency_key,
                            )
                        )
                        .mappings()
                        .first()
                    )
                    if previous:
                        if previous["fingerprint"] != fingerprint:
                            fail(
                                "invalid_arguments", "Invocation key reused with different content"
                            )
                        return ToolResult.model_validate(previous["result_json"])
                if call.name in ORDER_WRITES:
                    order = self._validate_write(connection, session_id, call.name, args)
                    connection.execute(
                        update(pending)
                        .where(pending.c.session_id == session_id, pending.c.status == "prepared")
                        .values(status="superseded")
                    )
                    action = PendingAction(
                        action_id=uuid.uuid4().hex,
                        tool_name=call.name,
                        arguments=args,
                        order_id=order["order_id"],
                        order_version=str(order["version"]),
                        summary=summarize_action(call.name, args),
                        expires_at=self.clock() + timedelta(minutes=15),
                    )
                    connection.execute(
                        insert(pending).values(
                            action_id=action.action_id,
                            session_id=session_id,
                            status="prepared",
                            payload=action.model_dump(mode="json"),
                        )
                    )
                    data = {"pending_action": action.model_dump(mode="json")}
                else:
                    data = self._read_or_handoff(connection, session_id, call.name, args)
                result = ToolResult.model_validate(
                    {"call_id": call.call_id, "status": "ok", "data": data}
                )
                if idempotency_key:
                    connection.execute(
                        insert(invocations).values(
                            session_id=session_id,
                            invocation_key=idempotency_key,
                            fingerprint=fingerprint,
                            result_json=result.model_dump(mode="json"),
                        )
                    )
            return result
        except BusinessFault as exc:
            return ToolResult(call_id=call.call_id, status="error", error=exc.error)
        except (ValidationError, ValueError, SyntaxError, InvalidOperation, ZeroDivisionError):
            return ToolResult(
                call_id=call.call_id,
                status="error",
                error=ToolError(
                    code="invalid_arguments",
                    message="Invalid tool arguments or arithmetic expression",
                ),
            )

    def _read_or_handoff(self, connection: Connection, sid: str, name: str, args: dict):
        if name.startswith("find_user_id_by_"):
            if name == "find_user_id_by_email":
                query = select(users.c.user_id).where(
                    func.lower(users.c.email) == args["email"].strip().lower()
                )
            else:
                query = select(users.c.user_id, users.c.address_json).where(
                    func.lower(users.c.first_name) == args["first_name"].strip().lower(),
                    func.lower(users.c.last_name) == args["last_name"].strip().lower(),
                )
            rows = list(connection.execute(query).mappings())
            if name.endswith("name_zip"):
                rows = [r for r in rows if r["address_json"]["zip"] == args["zip"]]
            if len(rows) != 1:
                fail("not_found_or_forbidden", "User not found or ambiguous")
            uid = rows[0]["user_id"]
            existing = self._session(connection, sid)["authenticated_user_id"]
            if existing is not None and existing != uid:
                fail("policy_denied", "A session cannot switch to another user")
            connection.execute(
                update(sessions)
                .where(sessions.c.session_id == sid)
                .values(authenticated_user_id=uid)
            )
            return {"user_id": uid}
        if name == "calculate":
            return {"result": safe_calculate(args["expression"])}
        uid = self._owner(connection, sid)
        if name == "get_order_details":
            return self._owned_order(connection, sid, args["order_id"])
        if name == "get_user_details":
            if args["user_id"] != uid:
                fail("not_found_or_forbidden", "User not found or unavailable")
            return self.store.user(connection, uid)
        if name == "get_product_details":
            row = (
                connection.execute(
                    select(products).where(products.c.product_id == args["product_id"])
                )
                .mappings()
                .first()
            )
            if row is None:
                fail("not_found_or_forbidden", "Product not found")
            return {
                "product_id": row["product_id"],
                "name": row["name"],
                "variants": {
                    r["item_id"]: {
                        "item_id": r["item_id"],
                        "options": r["options_json"],
                        "available": bool(r["available"]),
                        "price": money(r["price_cents"]),
                    }
                    for r in connection.execute(
                        select(variants).where(variants.c.product_id == args["product_id"])
                    ).mappings()
                },
            }
        if name == "transfer_to_human_agents":
            hid = uuid.uuid4().hex
            connection.execute(
                insert(handoffs).values(handoff_id=hid, session_id=sid, summary=args["summary"])
            )
            return {"handoff_id": hid, "status": "handoff_requested"}
        fail("invalid_arguments", "Unknown tool")

    def _validate_write(self, connection: Connection, sid: str, name: str, args: dict) -> dict:
        order = self._owned_order(connection, sid, args["order_id"])
        required = {"delivered"} if name == "return_delivered_order_items" else {"pending"}
        if name == "modify_pending_order_address":
            required.add("pending (item modified)")
        if order["status"] not in required:
            fail("invalid_state", "Order state does not permit this operation")
        if name == "cancel_pending_order" and args["reason"] not in {
            "no longer needed",
            "ordered by mistake",
        }:
            fail("policy_denied", "Cancellation reason is not allowed")
        if name == "return_delivered_order_items":
            available = Counter(i["item_id"] for i in order["items"])
            if Counter(args["item_ids"]) - available:
                fail("policy_denied", "Return items exceed the order contents")
            method = (
                connection.execute(
                    select(methods).where(
                        methods.c.user_id == order["user_id"],
                        methods.c.method_id == args["payment_method_id"],
                    )
                )
                .mappings()
                .first()
            )
            original = order["payment_history"][0]["payment_method_id"]
            if method is None or (
                method["source"] != "gift_card" and args["payment_method_id"] != original
            ):
                fail(
                    "policy_denied", "Refund must use the original method or an existing gift card"
                )
        return order

    def confirm(self, session_id: str, action_id: str, approved: bool) -> ToolResult:
        try:
            with self.store.transaction(immediate=True) as connection:
                self._owner(connection, session_id)
                row = (
                    connection.execute(
                        select(pending).where(
                            pending.c.action_id == action_id, pending.c.session_id == session_id
                        )
                    )
                    .mappings()
                    .first()
                )
                if row is None:
                    fail("confirmation_required", "No matching confirmation in this session")
                assert row is not None
                previous = connection.execute(
                    select(operations.c.result_json).where(
                        operations.c.action_id == action_id, operations.c.session_id == session_id
                    )
                ).first()
                if previous:
                    saved = ToolResult.model_validate(previous[0])
                    if isinstance(saved.data, dict) and saved.data.get("executed") != approved:
                        fail("policy_denied", "A recorded confirmation decision cannot be reversed")
                    return saved
                if row["status"] != "prepared":
                    fail("confirmation_stale", "Confirmation is no longer active")
                action = PendingAction.model_validate(row["payload"])
                if action.expires_at <= self.clock():
                    fail("confirmation_stale", "Confirmation expired")
                if not approved:
                    connection.execute(
                        update(pending)
                        .where(pending.c.action_id == action_id)
                        .values(status="rejected")
                    )
                    result = ToolResult(
                        call_id=action_id,
                        status="ok",
                        data={"executed": False, "reason": "user_declined"},
                    )
                    connection.execute(
                        insert(operations).values(
                            action_id=action_id,
                            session_id=session_id,
                            result_json=result.model_dump(mode="json"),
                        )
                    )
                    return result
                order = self._validate_write(
                    connection, session_id, action.tool_name, action.arguments
                )
                if str(order["version"]) != action.order_version:
                    fail("confirmation_stale", "Order changed; request a new confirmation")
                self._apply_write(connection, order, action)
                result = ToolResult(
                    call_id=action_id,
                    status="ok",
                    data={"executed": True, "order": self.store.order(connection, action.order_id)},
                )
                connection.execute(
                    insert(operations).values(
                        action_id=action_id,
                        session_id=session_id,
                        result_json=result.model_dump(mode="json"),
                    )
                )
                connection.execute(
                    update(pending)
                    .where(pending.c.action_id == action_id)
                    .values(status="consumed")
                )
            return result
        except BusinessFault as exc:
            return ToolResult(call_id=action_id, status="error", error=exc.error)

    def _apply_write(self, connection: Connection, order: dict, action: PendingAction) -> None:
        args: dict = dict(action.arguments)
        values = {"version": order["version"] + 1}
        extra = {
            k: v
            for k, v in order.items()
            if k
            not in {
                "order_id",
                "user_id",
                "status",
                "address",
                "version",
                "items",
                "fulfillments",
                "payment_history",
            }
        }
        if action.tool_name == "modify_pending_order_address":
            address = Address.model_validate({k: v for k, v in args.items() if k != "order_id"})
            values["address_json"] = address.model_dump()
        elif action.tool_name == "return_delivered_order_items":
            values["status"] = "return requested"
            extra.update(
                return_items=sorted(args["item_ids"]),
                return_payment_method_id=args["payment_method_id"],
            )
            values["extra_json"] = extra
        elif action.tool_name == "cancel_pending_order":
            values["status"] = "cancelled"
            extra["cancel_reason"] = args["reason"]
            values["extra_json"] = extra
            # 保留固定工具的模拟退款语义：原支付记录对应追加退款，礼品卡余额回补。
            for index, payment in enumerate(order["payment_history"]):
                amount = cents(payment["amount"])
                connection.execute(
                    insert(payments).values(
                        order_id=order["order_id"],
                        line_index=len(order["payment_history"]) + index,
                        transaction_type="refund",
                        amount_cents=amount,
                        payment_method_id=payment["payment_method_id"],
                    )
                )
                connection.execute(
                    update(methods)
                    .where(
                        methods.c.user_id == order["user_id"],
                        methods.c.method_id == payment["payment_method_id"],
                        methods.c.source == "gift_card",
                    )
                    .values(balance_cents=methods.c.balance_cents + amount)
                )
        else:
            fail("policy_denied", "Write operation not implemented")
        connection.execute(
            update(orders).where(orders.c.order_id == order["order_id"]).values(**values)
        )
