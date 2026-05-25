"""
Tests for slippage model implementations.

Covers ZeroSlippage, FixedSlippage, PercentSlippage, VolumeSlippage.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from backtester.execution.slippage_models import (
    FixedSlippage,
    PercentSlippage,
    VolumeSlippage,
    ZeroSlippage,
)
from backtester.interfaces import Bar, Order, OrderSide, OrderType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _order(side: OrderSide = OrderSide.BUY, quantity: float = 100.0) -> Order:
    return Order(symbol="AAPL", side=side, type=OrderType.MARKET, quantity=quantity)


def _bar(volume: float = 10_000.0, close: float = 100.0) -> Bar:
    return Bar(
        symbol="AAPL",
        timestamp=datetime(2024, 1, 2, tzinfo=timezone.utc),
        open=close,
        high=close,
        low=close,
        close=close,
        volume=volume,
    )


# ---------------------------------------------------------------------------
# ZeroSlippage
# ---------------------------------------------------------------------------


class TestZeroSlippage:
    def test_buy_unchanged(self) -> None:
        assert ZeroSlippage().adjust(100.0, _order(OrderSide.BUY), _bar()) == pytest.approx(100.0)

    def test_sell_unchanged(self) -> None:
        assert ZeroSlippage().adjust(100.0, _order(OrderSide.SELL), _bar()) == pytest.approx(100.0)

    def test_arbitrary_price_unchanged(self) -> None:
        assert ZeroSlippage().adjust(123.456, _order(), _bar()) == pytest.approx(123.456)


# ---------------------------------------------------------------------------
# FixedSlippage
# ---------------------------------------------------------------------------


class TestFixedSlippage:
    def test_buy_adds_amount(self) -> None:
        result = FixedSlippage(0.05).adjust(100.0, _order(OrderSide.BUY), _bar())
        assert result == pytest.approx(100.05)

    def test_sell_subtracts_amount(self) -> None:
        result = FixedSlippage(0.05).adjust(100.0, _order(OrderSide.SELL), _bar())
        assert result == pytest.approx(99.95)

    def test_zero_amount_is_unchanged(self) -> None:
        result = FixedSlippage(0.0).adjust(100.0, _order(), _bar())
        assert result == pytest.approx(100.0)

    def test_negative_amount_raises(self) -> None:
        with pytest.raises(ValueError):
            FixedSlippage(-1.0)


# ---------------------------------------------------------------------------
# PercentSlippage
# ---------------------------------------------------------------------------


class TestPercentSlippage:
    def test_buy_increases_price(self) -> None:
        result = PercentSlippage(0.01).adjust(100.0, _order(OrderSide.BUY), _bar())
        assert result == pytest.approx(101.0)

    def test_sell_decreases_price(self) -> None:
        result = PercentSlippage(0.01).adjust(100.0, _order(OrderSide.SELL), _bar())
        assert result == pytest.approx(99.0)

    def test_zero_pct_unchanged(self) -> None:
        result = PercentSlippage(0.0).adjust(100.0, _order(), _bar())
        assert result == pytest.approx(100.0)

    def test_negative_pct_raises(self) -> None:
        with pytest.raises(ValueError):
            PercentSlippage(-0.01)

    def test_scales_with_price(self) -> None:
        result = PercentSlippage(0.01).adjust(200.0, _order(OrderSide.BUY), _bar())
        assert result == pytest.approx(202.0)


# ---------------------------------------------------------------------------
# VolumeSlippage
# ---------------------------------------------------------------------------


class TestVolumeSlippage:
    def test_buy_increases_price(self) -> None:
        model = VolumeSlippage(pct=0.01)
        # qty=100, volume=10_000 → impact=0.01*100/10000=0.0001 → price*(1+0.0001)
        result = model.adjust(100.0, _order(OrderSide.BUY, quantity=100.0), _bar(volume=10_000.0))
        assert result == pytest.approx(100.0 * (1 + 0.0001))

    def test_sell_decreases_price(self) -> None:
        model = VolumeSlippage(pct=0.01)
        result = model.adjust(100.0, _order(OrderSide.SELL, quantity=100.0), _bar(volume=10_000.0))
        assert result == pytest.approx(100.0 * (1 - 0.0001))

    def test_clamp_at_five_times_pct(self) -> None:
        model = VolumeSlippage(pct=0.01)
        # qty=100_000, volume=100 → impact=0.01*1000=10, clamped to 5*0.01=0.05
        result = model.adjust(100.0, _order(OrderSide.BUY, quantity=100_000.0), _bar(volume=100.0))
        assert result == pytest.approx(100.0 * (1 + 0.05))

    def test_zero_volume_treated_as_one(self) -> None:
        model = VolumeSlippage(pct=0.01)
        # Should not raise even with zero volume
        result = model.adjust(100.0, _order(OrderSide.BUY, quantity=1.0), _bar(volume=0.0))
        assert result > 100.0

    def test_negative_pct_raises(self) -> None:
        with pytest.raises(ValueError):
            VolumeSlippage(-0.01)

    def test_zero_pct_unchanged(self) -> None:
        result = VolumeSlippage(0.0).adjust(100.0, _order(), _bar())
        assert result == pytest.approx(100.0)
