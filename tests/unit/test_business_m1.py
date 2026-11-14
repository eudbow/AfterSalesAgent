"""Database-state assertions for the four MVP business operations."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from sqlalchemy import func, select, update

from aftersalesagent.business.database import (
    RetailStore,
    cents,
    import_pinned_database,
    items,
    methods,
    money,
    operations,
    orders,
)
from aftersalesagent.business.service import RetailService, safe_calculate
from aftersalesagent.business.source_audit import load_snapshot
from aftersalesagent.schemas.contracts import RunContext, ToolCall

ROOT = Path(__file__).resolve().parents[2]
NOW = datetime(2026, 10, 16, 12, tzinfo=UTC)


def invoke(service: RetailService, sid: str, name: str, **kwargs):
    return service.dispatch(sid, ToolCall(call_id="test-call", name=name, arguments=kwargs))


def authenticated(service: RetailService, uid: str) -> str:
    sid = service.start(RunContext(environment_id=service.environment_id)).session_id
    with service.store.transaction() as connection:
        user = service.store.user(connection, uid)
    assert user is not None
    result = invoke(service, sid, "find_user_id_by_email", email=user["email"])
    assert result.status == "ok"
    return sid


def propose(service: RetailService, sid: str, name: str, **kwargs):
    result = invoke(service, sid, name, **kwargs)
    assert result.status == "ok", result
    action = service.view(sid).pending_action
    assert action is not None
    return action


def read_order(store: RetailStore, oid: str) -> dict:
    with store.transaction() as connection:
        order = store.order(connection, oid)
    assert order is not None
    return order


def test_import_roundtrip_preserves_business_facts(store: RetailStore) -> None:
    before = load_snapshot(ROOT).model_dump()
    after = store.export_snapshot().model_dump()
    # 用户订单列表没有时间语义；比较集合，商品行/付款列表仍严格保留顺序。
    for value in (before, after):
        for user in value["users"].values():
            user["orders"].sort()
    assert after == before


def test_import_refuses_to_replace_existing_database(store: RetailStore) -> None:
    before = store.path.read_bytes()
    with pytest.raises(FileExistsError):
        import_pinned_database(ROOT, store.path)
    assert store.path.read_bytes() == before


def test_money_is_exact_and_rejects_fractional_cent() -> None:
    assert cents(Decimal("0.29")) == 29
    assert money(29) == "0.29"
    with pytest.raises(ValueError):
        cents(Decimal("1.001"))


def test_environment_identity_is_bound_to_database(store: RetailStore) -> None:
    with pytest.raises(ValueError):
        RetailService(store).start(RunContext(environment_id="different_environment"))


def test_unauthenticated_order_lookup_has_no_data(store: RetailStore) -> None:
    service = RetailService(store)
    sid = service.start(RunContext(environment_id=service.environment_id)).session_id
    result = invoke(service, sid, "get_order_details", order_id="#W5056519")
    assert result.status == "error" and result.data is None


def test_user_id_alone_cannot_authenticate_and_session_cannot_switch(store: RetailStore) -> None:
    service = RetailService(store)
    sid = authenticated(service, "yara_muller_8652")
    result = invoke(service, sid, "find_user_id_by_email", email="mia.garcia2723@example.com")
    assert result.error is not None and result.error.code == "policy_denied"
    assert service.view(sid).authenticated_user_id == "yara_muller_8652"


def test_wrong_zip_can_be_corrected(store: RetailStore) -> None:
    service = RetailService(store)
    sid = service.start(RunContext(environment_id=service.environment_id)).session_id
    wrong = invoke(
        service, sid, "find_user_id_by_name_zip", first_name="Yara", last_name="Muller", zip="00000"
    )
    assert wrong.status == "error"
    right = invoke(
        service, sid, "find_user_id_by_name_zip", first_name="Yara", last_name="Muller", zip="85041"
    )
    assert right.status == "ok"


def test_foreign_and_nonexistent_orders_are_indistinguishable(store: RetailStore) -> None:
    service = RetailService(store)
    sid = authenticated(service, "yara_muller_8652")
    foreign = invoke(service, sid, "get_order_details", order_id="#W5490111")
    missing = invoke(service, sid, "get_order_details", order_id="#Wnot-real")
    assert foreign.error == missing.error
    assert foreign.data is missing.data is None


def test_cancel_requires_bound_confirmation_and_is_idempotent(store: RetailStore) -> None:
    service = RetailService(store, clock=lambda: NOW)
    sid = authenticated(service, "yara_muller_8652")
    before = read_order(store, "#W5056519")
    action = propose(
        service, sid, "cancel_pending_order", order_id="#W5056519", reason="ordered by mistake"
    )
    assert read_order(store, "#W5056519") == before
    no_action = service.confirm(sid, "invented-action-id", True)
    assert no_action.status == "error"
    first = service.confirm(sid, action.action_id, True)
    second = service.confirm(sid, action.action_id, True)
    assert first == second and first.status == "ok"
    after = read_order(store, "#W5056519")
    assert after["status"] == "cancelled"
    assert after["version"] == before["version"] + 1
    assert len(after["payment_history"]) == len(before["payment_history"]) + 1
    assert after["payment_history"][-1]["amount"] == before["payment_history"][0]["amount"]
    with store.transaction() as connection:
        assert connection.scalar(select(func.count()).select_from(operations)) == 1


def test_gift_card_cancel_refund_is_credited_once(store: RetailStore) -> None:
    snapshot = load_snapshot(ROOT)
    order = next(
        o
        for o in snapshot.orders.values()
        if o.status == "pending"
        and snapshot.users[o.user_id].payment_methods[o.payment_history[0].payment_method_id].source
        == "gift_card"
    )
    service = RetailService(store)
    sid = authenticated(service, order.user_id)
    pid = order.payment_history[0].payment_method_id
    with store.transaction() as connection:
        initial = connection.scalar(
            select(methods.c.balance_cents).where(
                methods.c.user_id == order.user_id, methods.c.method_id == pid
            )
        )
    action = propose(
        service, sid, "cancel_pending_order", order_id=order.order_id, reason="no longer needed"
    )
    service.confirm(sid, action.action_id, True)
    service.confirm(sid, action.action_id, True)
    with store.transaction() as connection:
        final = connection.scalar(
            select(methods.c.balance_cents).where(
                methods.c.user_id == order.user_id, methods.c.method_id == pid
            )
        )
    assert isinstance(initial, int) and isinstance(final, int)
    assert final - initial == sum(cents(p.amount) for p in order.payment_history)


@pytest.mark.parametrize("reason", ["wrong reason", ""])
def test_cancel_reason_is_validated(store: RetailStore, reason: str) -> None:
    service = RetailService(store)
    sid = authenticated(service, "yara_muller_8652")
    result = invoke(service, sid, "cancel_pending_order", order_id="#W5056519", reason=reason)
    assert result.status == "error"
    assert read_order(store, "#W5056519")["status"] == "pending"


def test_delivered_order_cannot_be_cancelled(store: RetailStore) -> None:
    service = RetailService(store)
    sid = authenticated(service, "mia_garcia_4516")
    result = invoke(
        service, sid, "cancel_pending_order", order_id="#W5490111", reason="no longer needed"
    )
    assert result.error is not None and result.error.code == "invalid_state"


def test_decline_and_supersede_do_not_change_orders(store: RetailStore) -> None:
    service = RetailService(store)
    sid = authenticated(service, "yara_muller_8652")
    first = propose(
        service, sid, "cancel_pending_order", order_id="#W5056519", reason="ordered by mistake"
    )
    assert service.confirm(sid, first.action_id, False).status == "ok"
    assert service.confirm(sid, first.action_id, True).status == "error"
    second = propose(
        service, sid, "cancel_pending_order", order_id="#W5056519", reason="no longer needed"
    )
    service.invalidate_pending(sid)
    assert service.confirm(sid, second.action_id, True).status == "error"
    assert read_order(store, "#W5056519")["status"] == "pending"


def test_confirmation_cannot_be_reused_in_another_session(store: RetailStore) -> None:
    service = RetailService(store)
    sid = authenticated(service, "yara_muller_8652")
    other = authenticated(service, "yara_muller_8652")
    action = propose(
        service, sid, "cancel_pending_order", order_id="#W5056519", reason="ordered by mistake"
    )
    result = service.confirm(other, action.action_id, True)
    assert result.status == "error"
    assert read_order(store, "#W5056519")["status"] == "pending"


def test_expired_confirmation_does_not_execute(store: RetailStore) -> None:
    now = [NOW]
    service = RetailService(store, clock=lambda: now[0])
    sid = authenticated(service, "yara_muller_8652")
    action = propose(
        service, sid, "cancel_pending_order", order_id="#W5056519", reason="ordered by mistake"
    )
    now[0] += timedelta(minutes=16)
    result = service.confirm(sid, action.action_id, True)
    assert result.error is not None and result.error.code == "confirmation_stale"
    assert read_order(store, "#W5056519")["status"] == "pending"


def test_order_version_change_invalidates_confirmation(store: RetailStore) -> None:
    service = RetailService(store)
    sid = authenticated(service, "yara_muller_8652")
    action = propose(
        service, sid, "cancel_pending_order", order_id="#W5056519", reason="ordered by mistake"
    )
    with store.transaction() as connection:
        connection.execute(update(orders).where(orders.c.order_id == "#W5056519").values(version=1))
    result = service.confirm(sid, action.action_id, True)
    assert result.error is not None and result.error.code == "confirmation_stale"
    assert read_order(store, "#W5056519")["status"] == "pending"


def test_address_change_preserves_items_prices_and_user_address(store: RetailStore) -> None:
    service = RetailService(store)
    sid = authenticated(service, "yusuf_taylor_7149")
    before = read_order(store, "#W2702727")
    with store.transaction() as connection:
        user_before = store.user(connection, "yusuf_taylor_7149")
    args = dict(
        order_id="#W2702727",
        address1="1234 Elm St",
        address2="",
        city="Springfield",
        state="IL",
        country="USA",
        zip="62701",
    )
    action = propose(service, sid, "modify_pending_order_address", **args)
    assert read_order(store, "#W2702727") == before
    service.confirm(sid, action.action_id, True)
    after = read_order(store, "#W2702727")
    assert after["address"]["zip"] == "62701"
    assert after["items"] == before["items"]
    assert after["payment_history"] == before["payment_history"]
    with store.transaction() as connection:
        assert store.user(connection, "yusuf_taylor_7149") == user_before


def test_return_updates_request_not_payment_and_cannot_repeat(store: RetailStore) -> None:
    service = RetailService(store)
    sid = authenticated(service, "mia_garcia_4516")
    before = read_order(store, "#W5490111")
    action = propose(
        service,
        sid,
        "return_delivered_order_items",
        order_id="#W5490111",
        item_ids=["1421289881"],
        payment_method_id="credit_card_3124723",
    )
    service.confirm(sid, action.action_id, True)
    after = read_order(store, "#W5490111")
    assert after["status"] == "return requested"
    assert after["return_items"] == ["1421289881"]
    assert after["payment_history"] == before["payment_history"]
    again = invoke(
        service,
        sid,
        "return_delivered_order_items",
        order_id="#W5490111",
        item_ids=["1421289881"],
        payment_method_id="credit_card_3124723",
    )
    assert again.status == "error"


def test_wrong_refund_method_or_empty_return_is_rejected(store: RetailStore) -> None:
    service = RetailService(store)
    sid = authenticated(service, "mia_garcia_4516")
    wrong = invoke(
        service,
        sid,
        "return_delivered_order_items",
        order_id="#W5490111",
        item_ids=["1421289881"],
        payment_method_id="paypal_9497703",
    )
    empty = invoke(
        service,
        sid,
        "return_delivered_order_items",
        order_id="#W5490111",
        item_ids=[],
        payment_method_id="credit_card_3124723",
    )
    assert wrong.error is not None and wrong.error.code == "policy_denied"
    assert empty.status == "error"


def test_return_checks_multiplicity_of_duplicate_items(store: RetailStore) -> None:
    snapshot = load_snapshot(ROOT)
    order = next(
        o
        for o in snapshot.orders.values()
        if o.status == "delivered" and len(o.items) != len({i.item_id for i in o.items})
    )
    counts = {}
    for item in order.items:
        counts[item.item_id] = counts.get(item.item_id, 0) + 1
    iid, count = next((k, v) for k, v in counts.items() if v > 1)
    service = RetailService(store)
    sid = authenticated(service, order.user_id)
    bad = invoke(
        service,
        sid,
        "return_delivered_order_items",
        order_id=order.order_id,
        item_ids=[iid] * (count + 1),
        payment_method_id=order.payment_history[0].payment_method_id,
    )
    assert bad.status == "error"
    good = propose(
        service,
        sid,
        "return_delivered_order_items",
        order_id=order.order_id,
        item_ids=[iid] * count,
        payment_method_id=order.payment_history[0].payment_method_id,
    )
    assert service.confirm(sid, good.action_id, True).status == "ok"
    with store.transaction() as connection:
        assert connection.scalar(
            select(func.count()).select_from(items).where(items.c.order_id == order.order_id)
        ) == len(order.items)


@pytest.mark.parametrize("expression", ["__import__('os')", "2**100", "3//2", "1/0", "1e99"])
def test_arithmetic_cannot_execute_code_or_unbounded_operations(expression: str) -> None:
    with pytest.raises((ValueError, SyntaxError, ArithmeticError)):
        safe_calculate(expression)
    assert safe_calculate("3131.10 + 4777.75 + 367.38") == "8276.23"


def test_failed_transaction_rolls_back_refund_and_order(store: RetailStore, monkeypatch) -> None:
    service = RetailService(store)
    sid = authenticated(service, "yara_muller_8652")
    before = read_order(store, "#W5056519")
    action = propose(
        service, sid, "cancel_pending_order", order_id="#W5056519", reason="ordered by mistake"
    )
    apply = service._apply_write

    def fail_after_write(*args):
        apply(*args)
        raise RuntimeError("Injected transaction failure")

    monkeypatch.setattr(service, "_apply_write", fail_after_write)
    with pytest.raises(RuntimeError):
        service.confirm(sid, action.action_id, True)
    assert read_order(store, "#W5056519") == before
    with store.transaction() as connection:
        assert connection.scalar(select(func.count()).select_from(operations)) == 0
    monkeypatch.setattr(service, "_apply_write", apply)
    assert service.confirm(sid, action.action_id, True).status == "ok"


def test_two_episodes_are_isolated(retail_template: Path, tmp_path: Path) -> None:
    from aftersalesagent.business.database import clone_database

    first = clone_database(retail_template, tmp_path / "a.sqlite")
    second = clone_database(retail_template, tmp_path / "b.sqlite")
    try:
        service = RetailService(first)
        sid = authenticated(service, "yara_muller_8652")
        action = propose(
            service, sid, "cancel_pending_order", order_id="#W5056519", reason="ordered by mistake"
        )
        service.confirm(sid, action.action_id, True)
        assert read_order(first, "#W5056519")["status"] == "cancelled"
        assert read_order(second, "#W5056519")["status"] == "pending"
    finally:
        first.close()
        second.close()


def test_tool_arguments_cannot_inject_identity_or_confirmation(store: RetailStore) -> None:
    service = RetailService(store)
    sid = authenticated(service, "yara_muller_8652")
    result = invoke(
        service,
        sid,
        "cancel_pending_order",
        order_id="#W5056519",
        reason="ordered by mistake",
        authenticated_user_id="mia_garcia_4516",
        approved=True,
    )
    assert result.status == "error"
    assert service.view(sid).pending_action is None
    assert read_order(store, "#W5056519")["status"] == "pending"


def test_pending_order_cannot_be_returned(store: RetailStore) -> None:
    service = RetailService(store)
    sid = authenticated(service, "yara_muller_8652")
    result = invoke(
        service,
        sid,
        "return_delivered_order_items",
        order_id="#W5056519",
        item_ids=["7902309762"],
        payment_method_id="credit_card_3095586",
    )
    assert result.error is not None and result.error.code == "invalid_state"
