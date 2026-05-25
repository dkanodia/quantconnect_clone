"""
Tests for execution model implementations.

Covers NextOpenExecution, SameBarExecution, VWAPExecution.
Each model is tested with ZeroSlippage + ZeroCommission for price correctness,
then with non-trivial slippage/commission models to verify the pipeline.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from backtester.execution.commission_models import FixedPerTrade, ZeroCommission
from backtester.execution.execution_models import (
    NextOpenExecution,
    SameBarExecution,
    VWAPExecution,
)
from backtester.execution.slippage_models import FixedSlippage, ZeroSlippage
from backtester.interfaces import Bar, Fill, Order, OrderSide, OrderType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TS = datetime(2024, 1, 2, tzinfo=timezone.utc)
_TS2 = datetime(2024, 1, 3, tzinfo=timezone.utc)


def _bar(
    ts: datetime = _TS,
    open_: float = 100.0,
    high: float = 110.0,
    low: float = 90.0,
    close: float = 105.0,
    volume: float = 10_000.0,
    symbol: str = "AAPL",
) -> Bar:
    return Bar(
        symbol=symbol,
        timestamp=ts,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def _order(side: OrderSide = OrderSide.BUY, qty: float = 10.0) -> Order:
    return Order(symbol="AAPL", side=side, type=OrderType.MARKET, quantity=qty)


def _zero_exec(model_cls):
    return model_cls(ZeroSlippage(), ZeroCommission())


# ---------------------------------------------------------------------------
# NextOpenExecution
# ---------------------------------------------------------------------------


class TestNextOpenExecution:
    def test_fills_at_next_bar_open(self) -> None:
        model = _zero_exec(NextOpenExecution)
        current = _bar(open_=100.0, close=105.0)
        next_ = _bar(ts=_TS2, open_=108.0, close=112.0)
        fill = model.execute(_order(), current, next_)
        assert fill is not None
        assert fill.price == pytest.approx(108.0)

    def test_returns_none_when_no_next_bar(self) -> None:
        model = _zero_exec(NextOpenExecution)
        fill = model.execute(_order(), _bar(), None)
        assert fill is None

    def test_fill_timestamp_is_next_bar(self) -> None:
        model = _zero_exec(NextOpenExecution)
        next_ = _bar(ts=_TS2, open_=110.0)
        fill = model.execute(_order(), _bar(), next_)
        assert fill is not None
        assert fill.timestamp == _TS2

    def test_fill_quantity_matches_order(self) -> None:
        model = _zero_exec(NextOpenExecution)
        fill = model.execute(_order(qty=25.0), _bar(), _bar(ts=_TS2))
        assert fill is not None
        assert fill.quantity == pytest.approx(25.0)

    def test_slippage_applied_to_next_open(self) -> None:
        model = NextOpenExecution(FixedSlippage(0.50), ZeroCommission())
        current = _bar(open_=100.0)
        next_ = _bar(ts=_TS2, open_=100.0)
        fill = model.execute(_order(OrderSide.BUY), current, next_)
        assert fill is not None
        assert fill.price == pytest.approx(100.50)

    def test_commission_computed_after_slippage(self) -> None:
        model = NextOpenExecution(ZeroSlippage(), FixedPerTrade(7.0))
        fill = model.execute(_order(), _bar(), _bar(ts=_TS2, open_=100.0))
        assert fill is not None
        assert fill.commission == pytest.approx(7.0)

    def test_fill_order_id_matches(self) -> None:
        model = _zero_exec(NextOpenExecution)
        order = _order()
        fill = model.execute(order, _bar(), _bar(ts=_TS2))
        assert fill is not None
        assert fill.order_id == order.id


# ---------------------------------------------------------------------------
# SameBarExecution
# ---------------------------------------------------------------------------


class TestSameBarExecution:
    def test_fills_at_current_bar_close(self) -> None:
        model = _zero_exec(SameBarExecution)
        current = _bar(close=105.0)
        fill = model.execute(_order(), current, None)
        assert fill is not None
        assert fill.price == pytest.approx(105.0)

    def test_fills_even_with_no_next_bar(self) -> None:
        model = _zero_exec(SameBarExecution)
        fill = model.execute(_order(), _bar(), None)
        assert fill is not None

    def test_fills_with_next_bar_present(self) -> None:
        model = _zero_exec(SameBarExecution)
        fill = model.execute(_order(), _bar(close=105.0), _bar(ts=_TS2, open_=120.0))
        assert fill is not None
        assert fill.price == pytest.approx(105.0)

    def test_fill_timestamp_is_current_bar(self) -> None:
        model = _zero_exec(SameBarExecution)
        current = _bar(ts=_TS, close=100.0)
        fill = model.execute(_order(), current, _bar(ts=_TS2))
        assert fill is not None
        assert fill.timestamp == _TS

    def test_slippage_on_sell_reduces_price(self) -> None:
        model = SameBarExecution(FixedSlippage(1.0), ZeroCommission())
        current = _bar(close=100.0)
        fill = model.execute(_order(OrderSide.SELL), current, None)
        assert fill is not None
        assert fill.price == pytest.approx(99.0)

    def test_commission_assigned(self) -> None:
        model = SameBarExecution(ZeroSlippage(), FixedPerTrade(3.0))
        fill = model.execute(_order(), _bar(), None)
        assert fill is not None
        assert fill.commission == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# VWAPExecution
# ---------------------------------------------------------------------------


class TestVWAPExecution:
    def test_fills_at_hlc3_of_next_bar(self) -> None:
        model = _zero_exec(VWAPExecution)
        next_ = _bar(ts=_TS2, high=110.0, low=90.0, close=100.0)
        fill = model.execute(_order(), _bar(), next_)
        assert fill is not None
        expected_vwap = (110.0 + 90.0 + 100.0) / 3
        assert fill.price == pytest.approx(expected_vwap)

    def test_returns_none_when_no_next_bar(self) -> None:
        model = _zero_exec(VWAPExecution)
        fill = model.execute(_order(), _bar(), None)
        assert fill is None

    def test_fill_timestamp_is_next_bar(self) -> None:
        model = _zero_exec(VWAPExecution)
        next_ = _bar(ts=_TS2, high=110.0, low=90.0, close=100.0)
        fill = model.execute(_order(), _bar(), next_)
        assert fill is not None
        assert fill.timestamp == _TS2

    def test_slippage_applied_to_vwap(self) -> None:
        model = VWAPExecution(FixedSlippage(1.0), ZeroCommission())
        next_ = _bar(ts=_TS2, high=90.0, low=90.0, close=90.0)  # vwap=90
        fill = model.execute(_order(OrderSide.BUY), _bar(), next_)
        assert fill is not None
        assert fill.price == pytest.approx(91.0)

    def test_commission_assigned(self) -> None:
        model = VWAPExecution(ZeroSlippage(), FixedPerTrade(5.0))
        fill = model.execute(_order(), _bar(), _bar(ts=_TS2))
        assert fill is not None
        assert fill.commission == pytest.approx(5.0)
