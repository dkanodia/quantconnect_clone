"""
Tests for commission model implementations.

Covers ZeroCommission, FixedPerTrade, PercentCommission, TieredCommission.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from backtester.execution.commission_models import (
    FixedPerTrade,
    PercentCommission,
    TieredCommission,
    ZeroCommission,
)
from backtester.interfaces import Fill, OrderSide


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _fill(price: float = 100.0, quantity: float = 10.0) -> Fill:
    return Fill(
        order_id="o1",
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=quantity,
        price=price,
        commission=0.0,
        timestamp=datetime(2024, 1, 2, tzinfo=timezone.utc),
    )


# ---------------------------------------------------------------------------
# ZeroCommission
# ---------------------------------------------------------------------------


class TestZeroCommission:
    def test_always_zero(self) -> None:
        assert ZeroCommission().compute(_fill(100.0, 10.0)) == 0.0

    def test_large_fill_still_zero(self) -> None:
        assert ZeroCommission().compute(_fill(500.0, 1000.0)) == 0.0


# ---------------------------------------------------------------------------
# FixedPerTrade
# ---------------------------------------------------------------------------


class TestFixedPerTrade:
    def test_returns_fixed_amount(self) -> None:
        assert FixedPerTrade(5.0).compute(_fill(100.0, 10.0)) == pytest.approx(5.0)

    def test_independent_of_price_and_quantity(self) -> None:
        model = FixedPerTrade(7.0)
        assert model.compute(_fill(100.0, 1.0)) == pytest.approx(7.0)
        assert model.compute(_fill(500.0, 1000.0)) == pytest.approx(7.0)

    def test_zero_commission_allowed(self) -> None:
        assert FixedPerTrade(0.0).compute(_fill()) == 0.0

    def test_negative_amount_raises(self) -> None:
        with pytest.raises(ValueError):
            FixedPerTrade(-1.0)


# ---------------------------------------------------------------------------
# PercentCommission
# ---------------------------------------------------------------------------


class TestPercentCommission:
    def test_basic_calculation(self) -> None:
        # price=100, qty=10, pct=0.001 → 100*10*0.001 = 1.0
        assert PercentCommission(0.001).compute(_fill(100.0, 10.0)) == pytest.approx(1.0)

    def test_scales_with_notional(self) -> None:
        model = PercentCommission(0.002)
        assert model.compute(_fill(200.0, 5.0)) == pytest.approx(200 * 5 * 0.002)

    def test_zero_pct_is_zero(self) -> None:
        assert PercentCommission(0.0).compute(_fill()) == pytest.approx(0.0)

    def test_negative_pct_raises(self) -> None:
        with pytest.raises(ValueError):
            PercentCommission(-0.001)


# ---------------------------------------------------------------------------
# TieredCommission
# ---------------------------------------------------------------------------


class TestTieredCommission:
    def test_lowest_tier_applied_below_first_threshold(self) -> None:
        # tiers: 0→0.002, 10_000→0.001
        # trade_value = 100*10 = 1000 < 10_000 → rate = 0.002
        model = TieredCommission([(0, 0.002), (10_000, 0.001)])
        assert model.compute(_fill(100.0, 10.0)) == pytest.approx(1000 * 0.002)

    def test_highest_applicable_tier_wins(self) -> None:
        # trade_value = 200*100 = 20_000 ≥ 10_000 → rate = 0.001
        model = TieredCommission([(0, 0.002), (10_000, 0.001)])
        assert model.compute(_fill(200.0, 100.0)) == pytest.approx(20_000 * 0.001)

    def test_three_tiers_correct_selection(self) -> None:
        model = TieredCommission([(0, 0.005), (5_000, 0.003), (50_000, 0.001)])
        # trade_value = 100 * 10 = 1000 → tier 0 (0.005)
        assert model.compute(_fill(100.0, 10.0)) == pytest.approx(1000 * 0.005)
        # trade_value = 100 * 100 = 10_000 → tier 5000 (0.003)
        assert model.compute(_fill(100.0, 100.0)) == pytest.approx(10_000 * 0.003)
        # trade_value = 100 * 1000 = 100_000 → tier 50000 (0.001)
        assert model.compute(_fill(100.0, 1000.0)) == pytest.approx(100_000 * 0.001)

    def test_tiers_sorted_regardless_of_input_order(self) -> None:
        # Declare in reverse order — should still work correctly.
        model = TieredCommission([(10_000, 0.001), (0, 0.002)])
        assert model.compute(_fill(100.0, 10.0)) == pytest.approx(1000 * 0.002)

    def test_empty_tiers_raises(self) -> None:
        with pytest.raises(ValueError):
            TieredCommission([])

    def test_trade_exactly_at_threshold(self) -> None:
        # trade_value = 100 * 100 = 10_000, threshold = 10_000 → matches
        model = TieredCommission([(0, 0.002), (10_000, 0.001)])
        assert model.compute(_fill(100.0, 100.0)) == pytest.approx(10_000 * 0.001)
