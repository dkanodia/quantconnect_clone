"""
Tests for risk model implementations.

Covers NoRisk, MaxDrawdownHalt, PositionSizeLimit.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from backtester.exceptions import MaxDrawdownExceeded, PositionSizeLimitExceeded
from backtester.interfaces import Bar, Fill, Order, OrderSide, OrderType
from backtester.risk.risk_models import MaxDrawdownHalt, NoRisk, PositionSizeLimit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bar(close: float = 100.0, symbol: str = "AAPL") -> Bar:
    return Bar(
        symbol=symbol,
        timestamp=datetime(2024, 1, 2, tzinfo=timezone.utc),
        open=close,
        high=close,
        low=close,
        close=close,
        volume=10_000.0,
    )


def _order(symbol: str = "AAPL", qty: float = 10.0) -> Order:
    return Order(symbol=symbol, side=OrderSide.BUY, type=OrderType.MARKET, quantity=qty)


def _fill(symbol: str = "AAPL", qty: float = 10.0, price: float = 100.0) -> Fill:
    return Fill(
        order_id="o1",
        symbol=symbol,
        side=OrderSide.BUY,
        quantity=qty,
        price=price,
        commission=0.0,
        timestamp=datetime(2024, 1, 2, tzinfo=timezone.utc),
    )


def _portfolio(equity: float = 100_000.0):
    """Return a mock PortfolioTracker with a fixed total equity."""
    p = MagicMock()
    p.get_total_equity.return_value = equity
    return p


# ---------------------------------------------------------------------------
# NoRisk
# ---------------------------------------------------------------------------


class TestNoRisk:
    def test_check_always_true(self) -> None:
        model = NoRisk()
        assert model.check(_order(), _portfolio(), _bar()) is True

    def test_on_fill_is_noop(self) -> None:
        NoRisk().on_fill(_fill(), _portfolio())  # must not raise

    def test_on_fill_called_multiple_times_no_error(self) -> None:
        model = NoRisk()
        for _ in range(10):
            model.on_fill(_fill(), _portfolio())


# ---------------------------------------------------------------------------
# MaxDrawdownHalt
# ---------------------------------------------------------------------------


class TestMaxDrawdownHalt:
    def test_no_drawdown_returns_true(self) -> None:
        model = MaxDrawdownHalt(limit=0.20)
        assert model.check(_order(), _portfolio(100_000.0), _bar()) is True

    def test_increasing_equity_returns_true(self) -> None:
        model = MaxDrawdownHalt(limit=0.20)
        model.check(_order(), _portfolio(100_000.0), _bar())
        assert model.check(_order(), _portfolio(110_000.0), _bar()) is True

    def test_drawdown_below_limit_returns_true(self) -> None:
        model = MaxDrawdownHalt(limit=0.20)
        model.check(_order(), _portfolio(100_000.0), _bar())  # set peak
        # 10 % drawdown < 20 % limit
        assert model.check(_order(), _portfolio(90_000.0), _bar()) is True

    def test_drawdown_above_limit_raises(self) -> None:
        model = MaxDrawdownHalt(limit=0.20)
        model.check(_order(), _portfolio(100_000.0), _bar())  # set peak to 100k
        with pytest.raises(MaxDrawdownExceeded) as exc_info:
            model.check(_order(), _portfolio(70_000.0), _bar())  # 30 % drawdown
        assert exc_info.value.current_drawdown == pytest.approx(0.30)
        assert exc_info.value.limit == pytest.approx(0.20)

    def test_peak_tracked_across_calls(self) -> None:
        model = MaxDrawdownHalt(limit=0.30)
        model.check(_order(), _portfolio(100_000.0), _bar())
        model.check(_order(), _portfolio(120_000.0), _bar())  # new peak
        # 30 % from 120k = 84k → should raise with 32% drawdown from 120k
        with pytest.raises(MaxDrawdownExceeded):
            model.check(_order(), _portfolio(80_000.0), _bar())

    def test_on_fill_updates_peak(self) -> None:
        model = MaxDrawdownHalt(limit=0.20)
        p = _portfolio(150_000.0)
        model.on_fill(_fill(), p)
        assert model._peak_equity == pytest.approx(150_000.0)

    def test_invalid_limit_raises(self) -> None:
        with pytest.raises(ValueError):
            MaxDrawdownHalt(limit=0.0)
        with pytest.raises(ValueError):
            MaxDrawdownHalt(limit=1.5)

    def test_limit_exactly_one_is_valid(self) -> None:
        model = MaxDrawdownHalt(limit=1.0)
        # 100 % drawdown (equity=0) should raise since drawdown==1.0 > limit
        # Actually drawdown=1.0 which is not > 1.0, so it should NOT raise
        model.check(_order(), _portfolio(100_000.0), _bar())
        # Even if equity hits 0, drawdown=1.0 which is equal to limit, not >
        assert model.check(_order(), _portfolio(0.0), _bar()) is True


# ---------------------------------------------------------------------------
# PositionSizeLimit
# ---------------------------------------------------------------------------


class TestPositionSizeLimit:
    def test_within_limit_returns_true(self) -> None:
        # order_value=100*10=1000, equity=100_000, pct=0.01 → 1% ≤ 10%
        model = PositionSizeLimit(limit_pct=0.10)
        assert model.check(_order(qty=10.0), _portfolio(100_000.0), _bar(close=100.0)) is True

    def test_exceeds_limit_raises(self) -> None:
        # order_value=100*200=20_000, equity=100_000, pct=0.20 > 0.10
        model = PositionSizeLimit(limit_pct=0.10)
        with pytest.raises(PositionSizeLimitExceeded) as exc_info:
            model.check(_order(qty=200.0), _portfolio(100_000.0), _bar(close=100.0))
        assert exc_info.value.limit_pct == pytest.approx(0.10)
        assert exc_info.value.requested_pct == pytest.approx(0.20)

    def test_exactly_at_limit_is_allowed(self) -> None:
        # order_value=100*100=10_000, equity=100_000, pct=0.10 = limit
        model = PositionSizeLimit(limit_pct=0.10)
        assert model.check(_order(qty=100.0), _portfolio(100_000.0), _bar(close=100.0)) is True

    def test_zero_equity_always_passes(self) -> None:
        model = PositionSizeLimit(limit_pct=0.05)
        assert model.check(_order(qty=100.0), _portfolio(0.0), _bar()) is True

    def test_on_fill_noop(self) -> None:
        PositionSizeLimit(0.10).on_fill(_fill(), _portfolio())  # must not raise

    def test_invalid_limit_pct_raises(self) -> None:
        with pytest.raises(ValueError):
            PositionSizeLimit(0.0)
        with pytest.raises(ValueError):
            PositionSizeLimit(1.5)
