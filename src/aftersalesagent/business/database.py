"""SQLite schema and lossless Retail import. Business services own all mutations."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from decimal import Decimal
from pathlib import Path

from sqlalchemy import (
    JSON,
    Column,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    event,
    insert,
    select,
)
from sqlalchemy.engine import Connection

from aftersalesagent.business.source_audit import audit_relations, load_snapshot, verify_source
from aftersalesagent.schemas.retail import RetailSnapshot

metadata = MetaData()
users = Table(
    "users",
    metadata,
    Column("user_id", String, primary_key=True),
    Column("first_name", String, nullable=False),
    Column("last_name", String, nullable=False),
    Column("email", String, nullable=False),
    Column("address_json", JSON, nullable=False),
)
products = Table(
    "products",
    metadata,
    Column("product_id", String, primary_key=True),
    Column("name", String, nullable=False),
)
variants = Table(
    "product_variants",
    metadata,
    Column("product_id", ForeignKey("products.product_id"), primary_key=True),
    Column("item_id", String, primary_key=True),
    Column("options_json", JSON, nullable=False),
    Column("available", Integer, nullable=False),
    Column("price_cents", Integer, nullable=False),
)
methods = Table(
    "payment_methods",
    metadata,
    Column("user_id", ForeignKey("users.user_id"), primary_key=True),
    Column("method_id", String, primary_key=True),
    Column("source", String, nullable=False),
    Column("details_json", JSON, nullable=False),
    Column("balance_cents", Integer),
)
orders = Table(
    "orders",
    metadata,
    Column("order_id", String, primary_key=True),
    Column("user_id", ForeignKey("users.user_id"), nullable=False),
    Column("status", String, nullable=False),
    Column("address_json", JSON, nullable=False),
    Column("version", Integer, nullable=False),
    Column("extra_json", JSON, nullable=False),
)
items = Table(
    "order_items",
    metadata,
    Column("order_id", ForeignKey("orders.order_id"), primary_key=True),
    Column("line_index", Integer, primary_key=True),
    Column("product_id", String, nullable=False),
    Column("item_id", String, nullable=False),
    Column("name", String, nullable=False),
    Column("options_json", JSON, nullable=False),
    Column("purchase_price_cents", Integer, nullable=False),
)
fulfillments = Table(
    "order_fulfillments",
    metadata,
    Column("order_id", ForeignKey("orders.order_id"), primary_key=True),
    Column("line_index", Integer, primary_key=True),
    Column("payload", JSON, nullable=False),
)
payments = Table(
    "payment_history",
    metadata,
    Column("order_id", ForeignKey("orders.order_id"), primary_key=True),
    Column("line_index", Integer, primary_key=True),
    Column("transaction_type", String, nullable=False),
    Column("amount_cents", Integer, nullable=False),
    Column("payment_method_id", String, nullable=False),
)
sessions = Table(
    "sessions",
    metadata,
    Column("session_id", String, primary_key=True),
    Column("environment_id", String, nullable=False),
    Column("authenticated_user_id", String),
    Column("context_json", JSON, nullable=False),
)
pending = Table(
    "pending_actions",
    metadata,
    Column("action_id", String, primary_key=True),
    Column("session_id", ForeignKey("sessions.session_id"), nullable=False),
    Column("status", String, nullable=False),
    Column("payload", JSON, nullable=False),
)
operations = Table(
    "operation_records",
    metadata,
    Column("action_id", ForeignKey("pending_actions.action_id"), primary_key=True),
    Column("session_id", String, nullable=False),
    Column("result_json", JSON, nullable=False),
)
handoffs = Table(
    "handoffs",
    metadata,
    Column("handoff_id", String, primary_key=True),
    Column("session_id", String, nullable=False),
    Column("summary", String, nullable=False),
)
invocations = Table(
    "tool_invocations",
    metadata,
    Column("session_id", ForeignKey("sessions.session_id"), primary_key=True),
    Column("invocation_key", String, primary_key=True),
    Column("fingerprint", String, nullable=False),
    Column("result_json", JSON, nullable=False),
)
meta = Table(
    "environment_meta",
    metadata,
    Column("key", String, primary_key=True),
    Column("value", String, nullable=False),
)


def cents(value: Decimal | str | int) -> int:
    amount = Decimal(str(value)) * 100
    if not amount.is_finite() or amount != amount.to_integral_value():
        raise ValueError("Money must be finite and have at most two decimal places")
    return int(amount)


def money(value: int) -> str:
    return format(Decimal(value) / 100, ".2f")


class RetailStore:
    def __init__(self, path: Path):
        self.path = path.resolve()
        self.engine = create_engine(
            "sqlite+pysqlite:///" + self.path.as_posix(),
            connect_args={"check_same_thread": False, "timeout": 5},
        )

        @event.listens_for(self.engine, "connect")
        def enable_foreign_keys(dbapi_connection, _record):
            dbapi_connection.execute("PRAGMA foreign_keys=ON")

    @contextmanager
    def transaction(self, *, immediate: bool = False) -> Iterator[Connection]:
        with self.engine.begin() as connection:
            # 写操作先取得SQLite写锁，读取确认版本与提交变化在同一个事务内。
            if immediate:
                connection.exec_driver_sql("BEGIN IMMEDIATE")
            yield connection

    def close(self) -> None:
        self.engine.dispose()

    def order(self, connection: Connection, order_id: str) -> dict | None:
        row = (
            connection.execute(select(orders).where(orders.c.order_id == order_id))
            .mappings()
            .first()
        )
        if row is None:
            return None
        result = dict(row["extra_json"])
        result.update(
            order_id=row["order_id"],
            user_id=row["user_id"],
            status=row["status"],
            address=row["address_json"],
            version=row["version"],
        )
        result["items"] = [
            {
                "name": r["name"],
                "product_id": r["product_id"],
                "item_id": r["item_id"],
                "price": money(r["purchase_price_cents"]),
                "options": r["options_json"],
            }
            for r in connection.execute(
                select(items).where(items.c.order_id == order_id).order_by(items.c.line_index)
            ).mappings()
        ]
        result["fulfillments"] = [
            r["payload"]
            for r in connection.execute(
                select(fulfillments)
                .where(fulfillments.c.order_id == order_id)
                .order_by(fulfillments.c.line_index)
            ).mappings()
        ]
        result["payment_history"] = [
            {
                "transaction_type": r["transaction_type"],
                "amount": money(r["amount_cents"]),
                "payment_method_id": r["payment_method_id"],
            }
            for r in connection.execute(
                select(payments)
                .where(payments.c.order_id == order_id)
                .order_by(payments.c.line_index)
            ).mappings()
        ]
        return result

    def user(self, connection: Connection, user_id: str) -> dict | None:
        row = connection.execute(select(users).where(users.c.user_id == user_id)).mappings().first()
        if row is None:
            return None
        payment_methods = {}
        for r in connection.execute(select(methods).where(methods.c.user_id == user_id)).mappings():
            method = dict(r["details_json"])
            if r["balance_cents"] is not None:
                method["balance"] = money(r["balance_cents"])
            payment_methods[r["method_id"]] = method
        return {
            "user_id": user_id,
            "name": {"first_name": row["first_name"], "last_name": row["last_name"]},
            "email": row["email"],
            "address": row["address_json"],
            "payment_methods": payment_methods,
            "orders": list(
                connection.execute(
                    select(orders.c.order_id).where(orders.c.user_id == user_id)
                ).scalars()
            ),
        }

    def export_snapshot(self) -> RetailSnapshot:
        with self.transaction() as connection:
            user_map = {
                uid: self.user(connection, uid)
                for uid in connection.execute(select(users.c.user_id)).scalars()
            }
            order_map = {}
            for oid in connection.execute(select(orders.c.order_id)).scalars():
                order = self.order(connection, oid)
                assert order is not None
                order.pop("version")
                order_map[oid] = order
            product_map = {}
            for row in connection.execute(select(products)).mappings():
                pid = row["product_id"]
                product_map[pid] = {
                    "product_id": pid,
                    "name": row["name"],
                    "variants": {
                        r["item_id"]: {
                            "item_id": r["item_id"],
                            "options": r["options_json"],
                            "available": bool(r["available"]),
                            "price": money(r["price_cents"]),
                        }
                        for r in connection.execute(
                            select(variants).where(variants.c.product_id == pid)
                        ).mappings()
                    },
                }
        return RetailSnapshot.model_validate(
            {"users": user_map, "orders": order_map, "products": product_map}
        )


def initialize_database(path: Path, snapshot: RetailSnapshot, source_commit: str) -> RetailStore:
    errors = audit_relations(snapshot)
    if errors:
        raise ValueError("; ".join(errors))
    # 独占创建文件，避免误覆盖已有实验或运行数据。重置使用新的episode文件。
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb"):
        pass
    store = RetailStore(path)
    try:
        metadata.create_all(store.engine)
        with store.transaction() as connection:
            for user in snapshot.users.values():
                connection.execute(
                    insert(users).values(
                        user_id=user.user_id,
                        first_name=user.name.first_name,
                        last_name=user.name.last_name,
                        email=user.email,
                        address_json=user.address.model_dump(),
                    )
                )
                for method in user.payment_methods.values():
                    payload = json.loads(method.model_dump_json(exclude_none=True))
                    connection.execute(
                        insert(methods).values(
                            user_id=user.user_id,
                            method_id=method.id,
                            source=method.source,
                            details_json=payload,
                            balance_cents=None if method.balance is None else cents(method.balance),
                        )
                    )
            for product in snapshot.products.values():
                connection.execute(
                    insert(products).values(product_id=product.product_id, name=product.name)
                )
                for variant in product.variants.values():
                    connection.execute(
                        insert(variants).values(
                            product_id=product.product_id,
                            item_id=variant.item_id,
                            options_json=variant.options,
                            available=int(variant.available),
                            price_cents=cents(variant.price),
                        )
                    )
            for order in snapshot.orders.values():
                payload = json.loads(order.model_dump_json())
                extra = {
                    k: v
                    for k, v in payload.items()
                    if k
                    not in {
                        "order_id",
                        "user_id",
                        "status",
                        "address",
                        "items",
                        "fulfillments",
                        "payment_history",
                    }
                }
                connection.execute(
                    insert(orders).values(
                        order_id=order.order_id,
                        user_id=order.user_id,
                        status=order.status,
                        address_json=order.address.model_dump(),
                        version=0,
                        extra_json=extra,
                    )
                )
                for index, item in enumerate(order.items):
                    connection.execute(
                        insert(items).values(
                            order_id=order.order_id,
                            line_index=index,
                            product_id=item.product_id,
                            item_id=item.item_id,
                            name=item.name,
                            options_json=item.options,
                            purchase_price_cents=cents(item.price),
                        )
                    )
                for index, fulfillment in enumerate(order.fulfillments):
                    connection.execute(
                        insert(fulfillments).values(
                            order_id=order.order_id,
                            line_index=index,
                            payload=fulfillment.model_dump(),
                        )
                    )
                for index, payment in enumerate(order.payment_history):
                    connection.execute(
                        insert(payments).values(
                            order_id=order.order_id,
                            line_index=index,
                            transaction_type=payment.transaction_type,
                            amount_cents=cents(payment.amount),
                            payment_method_id=payment.payment_method_id,
                        )
                    )
            connection.execute(
                insert(meta),
                [
                    {"key": "source_commit", "value": source_commit},
                    {"key": "schema_version", "value": "M1.1"},
                ],
            )
        return store
    except Exception:
        # 不删除失败文件；下次使用新路径，避免将部分初始化误认为有效环境。
        store.close()
        raise


def import_pinned_database(root: Path, output: Path) -> RetailStore:
    errors = verify_source(root)
    if errors:
        raise ValueError("; ".join(errors))
    manifest = json.loads((root / "data/manifests/source_manifest.json").read_text("utf-8"))
    return initialize_database(output, load_snapshot(root), manifest["commit"])


def clone_database(template: Path, output: Path) -> RetailStore:
    """通过SQLite只读backup创建独立episode，不复制可能正写入的数据库文件。"""
    template, output = template.resolve(), output.resolve()
    if not template.is_file():
        raise FileNotFoundError(template)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb"):
        pass
    with closing(sqlite3.connect(template.as_uri() + "?mode=ro", uri=True)) as source:
        with closing(sqlite3.connect(output)) as target:
            source.backup(target)
    return RetailStore(output)
