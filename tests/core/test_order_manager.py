"""
Tests for OrderManager.

Covers:
- submit / cancel / record_fill / get_pending / get_fills / get_order
- Status transitions and error conditions
- State isolation between operations
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from backtester.core.order_manager import OrderManager
from backtester.exceptions import OrderError
from backtester.interfaces import Fill, Order, OrderSide, OrderStatus, OrderType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _order(
    symbol: str = "AAPL",
    side: OrderSide = OrderSide.BUY,
    quantity: float = 10.0,
) -> Order:
    return Order(symbol=symbol, side=side, type=OrderType.MARKET, quantity=quantity)


def _fill(order: Order, price: float = 100.0, commission: float = 1.0) -> Fill:
    return Fill(
        order_id=order.id,
        symbol=order.symbol,
        side=order.side,
        quantity=order.quantity,
        price=price,
        commission=commission,
        timestamp=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------


class TestSubmit:
    def test_submit_sets_status_to_submitted(self) -> None:
        om = OrderManager()
        order = _order()
        om.submit(order)
        assert order.status == OrderStatus.SUBMITTED

    def test_submit_duplicate_id_raises(self) -> None:
        om = OrderManager()
        order = _order()
        om.submit(order)
        with pytest.raises(OrderError, match=order.id):
            om.submit(order)

    def test_get_order_after_submit(self) -> None:
        om = OrderManager()
        order = _order()
        om.submit(order)
        assert om.get_order(order.id) is order

    def test_multiple_distinct_orders_all_pending(self) -> None:
        om = OrderManager()
        orders = [_order() for _ in range(3)]
        for o in orders:
            om.submit(o)
        assert len(om.get_pending()) == 3


# ---------------------------------------------------------------------------
# cancel
# ---------------------------------------------------------------------------


class TestCancel:
    def test_cancel_sets_status_cancelled(self) -> None:
        om = OrderManager()
        order = _order()
        om.submit(order)
        om.cancel(order.id)
        assert order.status == OrderStatus.CANCELLED

    def test_cancel_removes_from_pending(self) -> None:
        om = OrderManager()
        order = _order()
        om.submit(order)
        om.cancel(order.id)
        assert order not in om.get_pending()

    def test_cancel_unknown_id_raises(self) -> None:
        om = OrderManager()
        with pytest.raises(OrderError):
            om.cancel("does-not-exist")

    def test_cancel_filled_order_raises(self) -> None:
        om = OrderManager()
        order = _order()
        om.submit(order)
        om.record_fill(_fill(order))
        with pytest.raises(OrderError):
            om.cancel(order.id)

    def test_cancel_already_cancelled_raises(self) -> None:
        om = OrderManager()
        order = _order()
        om.submit(order)
        om.cancel(order.id)
        with pytest.raises(OrderError):
            om.cancel(order.id)


# ---------------------------------------------------------------------------
# record_fill
# ---------------------------------------------------------------------------


class TestRecordFill:
    def test_record_fill_sets_status_filled(self) -> None:
        om = OrderManager()
        order = _order()
        om.submit(order)
        om.record_fill(_fill(order))
        assert order.status == OrderStatus.FILLED

    def test_record_fill_stored_in_fills(self) -> None:
        om = OrderManager()
        order = _order()
        om.submit(order)
        fill = _fill(order)
        om.record_fill(fill)
        assert fill in om.get_fills()

    def test_record_fill_unknown_order_raises(self) -> None:
        om = OrderManager()
        fake_order = _order()
        fill = _fill(fake_order)
        with pytest.raises(OrderError):
            om.record_fill(fill)

    def test_filled_order_removed_from_pending(self) -> None:
        om = OrderManager()
        order = _order()
        om.submit(order)
        om.record_fill(_fill(order))
        assert order not in om.get_pending()

    def test_multiple_fills_accumulate(self) -> None:
        om = OrderManager()
        orders = [_order() for _ in range(3)]
        for o in orders:
            om.submit(o)
            om.record_fill(_fill(o))
        assert len(om.get_fills()) == 3


# ---------------------------------------------------------------------------
# get_pending / get_order / get_fills
# ---------------------------------------------------------------------------


class TestQueries:
    def test_get_pending_empty_initially(self) -> None:
        assert OrderManager().get_pending() == []

    def test_get_fills_empty_initially(self) -> None:
        assert OrderManager().get_fills() == []

    def test_get_order_unknown_raises(self) -> None:
        with pytest.raises(OrderError):
            OrderManager().get_order("nope")

    def test_pending_excludes_cancelled_and_filled(self) -> None:
        om = OrderManager()
        kept = _order()
        cancelled = _order()
        filled = _order()

        om.submit(kept)
        om.submit(cancelled)
        om.submit(filled)

        om.cancel(cancelled.id)
        om.record_fill(_fill(filled))

        assert om.get_pending() == [kept]

    def test_get_fills_returns_copy(self) -> None:
        om = OrderManager()
        original = om.get_fills()
        original.append("mutant")  # type: ignore[arg-type]
        assert len(om.get_fills()) == 0
