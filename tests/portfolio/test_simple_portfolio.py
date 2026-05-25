"""
Tests for SimplePortfolio.

Covers:
- Initial state
- apply_fill: BUY path (position creation, avg-cost update, cash deduction)
- apply_fill: SELL path (position reduction, realised PnL, trade records)
- apply_fill error conditions (InsufficientFunds, InsufficientPosition, NegativeCash)
- update_market (unrealised PnL)
- record_equity / get_equity_curve
- get_total_equity
- get_trades
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from backtester.exceptions import (
    InsufficientFundsError,
    InsufficientPositionError,
    NegativeCashError,
)
from backtester.interfaces import Bar, Fill, OrderSide
from backtester.portfolio.simple_portfolio import SimplePortfolio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TS = datetime(2024, 1, 2, tzinfo=timezone.utc)
_TS2 = datetime(2024, 1, 3, tzinfo=timezone.utc)


def _buy(symbol: str, qty: float, price: float, commission: float = 0.0) -> Fill:
    return Fill(
        order_id="buy-1",
        symbol=symbol,
        side=OrderSide.BUY,
        quantity=qty,
        price=price,
        commission=commission,
        timestamp=_TS,
    )


def _sell(symbol: str, qty: float, price: float, commission: float = 0.0) -> Fill:
    return Fill(
        order_id="sell-1",
        symbol=symbol,
        side=OrderSide.SELL,
        quantity=qty,
        price=price,
        commission=commission,
        timestamp=_TS2,
    )


def _bar(symbol: str, close: float) -> Bar:
    return Bar(
        symbol=symbol,
        timestamp=_TS,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1000.0,
    )


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


class TestInitialState:
    def test_default_initial_cash(self) -> None:
        p = SimplePortfolio()
        assert p.get_cash() == 100_000.0

    def test_custom_initial_cash(self) -> None:
        p = SimplePortfolio(initial_cash=50_000.0)
        assert p.get_cash() == 50_000.0

    def test_no_positions_initially(self) -> None:
        assert SimplePortfolio().get_positions() == {}

    def test_total_equity_equals_cash_initially(self) -> None:
        p = SimplePortfolio(initial_cash=10_000.0)
        assert p.get_total_equity() == 10_000.0

    def test_equity_curve_empty_initially(self) -> None:
        curve = SimplePortfolio().get_equity_curve()
        assert isinstance(curve, pd.Series)
        assert len(curve) == 0

    def test_trades_empty_initially(self) -> None:
        trades = SimplePortfolio().get_trades()
        assert isinstance(trades, pd.DataFrame)
        assert len(trades) == 0


# ---------------------------------------------------------------------------
# BUY fills
# ---------------------------------------------------------------------------


class TestBuyFills:
    def test_buy_reduces_cash(self) -> None:
        p = SimplePortfolio(10_000.0)
        p.apply_fill(_buy("AAPL", qty=10, price=100.0))
        assert p.get_cash() == pytest.approx(9_000.0)

    def test_buy_with_commission_reduces_cash(self) -> None:
        p = SimplePortfolio(10_000.0)
        p.apply_fill(_buy("AAPL", qty=10, price=100.0, commission=5.0))
        assert p.get_cash() == pytest.approx(8_995.0)

    def test_buy_creates_position(self) -> None:
        p = SimplePortfolio()
        p.apply_fill(_buy("AAPL", qty=10, price=100.0))
        assert "AAPL" in p.get_positions()
        pos = p.get_positions()["AAPL"]
        assert pos.quantity == 10.0
        assert pos.avg_entry_price == 100.0

    def test_buy_updates_avg_entry_price(self) -> None:
        p = SimplePortfolio()
        p.apply_fill(_buy("AAPL", qty=10, price=100.0))
        p.apply_fill(_buy("AAPL", qty=10, price=110.0))
        pos = p.get_positions()["AAPL"]
        assert pos.quantity == 20.0
        assert pos.avg_entry_price == pytest.approx(105.0)

    def test_buy_insufficient_funds_raises(self) -> None:
        p = SimplePortfolio(initial_cash=500.0)
        with pytest.raises(InsufficientFundsError) as exc_info:
            p.apply_fill(_buy("AAPL", qty=10, price=100.0))
        assert exc_info.value.required == pytest.approx(1000.0)
        assert exc_info.value.available == pytest.approx(500.0)

    def test_buy_exact_cash_is_allowed(self) -> None:
        p = SimplePortfolio(initial_cash=1000.0)
        p.apply_fill(_buy("AAPL", qty=10, price=100.0))
        assert p.get_cash() == pytest.approx(0.0)

    def test_buy_does_not_create_trade_record(self) -> None:
        p = SimplePortfolio()
        p.apply_fill(_buy("AAPL", qty=5, price=50.0))
        assert len(p.get_trades()) == 0


# ---------------------------------------------------------------------------
# SELL fills
# ---------------------------------------------------------------------------


class TestSellFills:
    def test_sell_increases_cash(self) -> None:
        p = SimplePortfolio(10_000.0)
        p.apply_fill(_buy("AAPL", qty=10, price=100.0))
        p.apply_fill(_sell("AAPL", qty=10, price=110.0))
        assert p.get_cash() == pytest.approx(10_100.0)  # 9000 + 1100

    def test_sell_records_trade(self) -> None:
        p = SimplePortfolio()
        p.apply_fill(_buy("AAPL", qty=10, price=100.0))
        p.apply_fill(_sell("AAPL", qty=10, price=120.0))
        trades = p.get_trades()
        assert len(trades) == 1
        row = trades.iloc[0]
        assert row["symbol"] == "AAPL"
        assert row["pnl"] == pytest.approx(200.0)

    def test_sell_realised_pnl_loss(self) -> None:
        p = SimplePortfolio()
        p.apply_fill(_buy("AAPL", qty=10, price=100.0))
        p.apply_fill(_sell("AAPL", qty=10, price=80.0))
        trades = p.get_trades()
        assert trades.iloc[0]["pnl"] == pytest.approx(-200.0)

    def test_sell_commission_deducted_from_proceeds(self) -> None:
        p = SimplePortfolio(10_000.0)
        p.apply_fill(_buy("AAPL", qty=10, price=100.0))
        p.apply_fill(_sell("AAPL", qty=10, price=100.0, commission=10.0))
        assert p.get_cash() == pytest.approx(9_990.0)  # 9000 + 1000 - 10

    def test_sell_partial_reduces_position(self) -> None:
        p = SimplePortfolio()
        p.apply_fill(_buy("AAPL", qty=10, price=100.0))
        p.apply_fill(_sell("AAPL", qty=4, price=100.0))
        pos = p.get_positions()["AAPL"]
        assert pos.quantity == pytest.approx(6.0)

    def test_sell_full_removes_position(self) -> None:
        p = SimplePortfolio()
        p.apply_fill(_buy("AAPL", qty=10, price=100.0))
        p.apply_fill(_sell("AAPL", qty=10, price=100.0))
        assert "AAPL" not in p.get_positions()

    def test_sell_more_than_held_raises(self) -> None:
        p = SimplePortfolio()
        p.apply_fill(_buy("AAPL", qty=10, price=100.0))
        with pytest.raises(InsufficientPositionError) as exc_info:
            p.apply_fill(_sell("AAPL", qty=11, price=100.0))
        assert exc_info.value.requested == pytest.approx(11.0)
        assert exc_info.value.held == pytest.approx(10.0)

    def test_sell_without_position_raises(self) -> None:
        p = SimplePortfolio()
        with pytest.raises(InsufficientPositionError):
            p.apply_fill(_sell("AAPL", qty=1, price=100.0))

    def test_multiple_partial_sells_record_multiple_trades(self) -> None:
        p = SimplePortfolio()
        p.apply_fill(_buy("AAPL", qty=10, price=100.0))
        p.apply_fill(_sell("AAPL", qty=5, price=110.0))
        p.apply_fill(_sell("AAPL", qty=5, price=120.0))
        assert len(p.get_trades()) == 2


# ---------------------------------------------------------------------------
# update_market
# ---------------------------------------------------------------------------


class TestUpdateMarket:
    def test_update_market_sets_unrealised_pnl(self) -> None:
        p = SimplePortfolio()
        p.apply_fill(_buy("AAPL", qty=10, price=100.0))
        p.update_market([_bar("AAPL", close=110.0)])
        pos = p.get_positions()["AAPL"]
        assert pos.unrealized_pnl == pytest.approx(100.0)

    def test_update_market_updates_last_price(self) -> None:
        p = SimplePortfolio()
        p.apply_fill(_buy("AAPL", qty=10, price=100.0))
        p.update_market([_bar("AAPL", close=150.0)])
        assert p.get_total_equity() == pytest.approx(
            p.get_cash() + 10 * 150.0
        )

    def test_update_market_ignores_unknown_symbols(self) -> None:
        p = SimplePortfolio()
        p.update_market([_bar("UNKNOWN", close=999.0)])  # should not raise

    def test_update_market_multiple_bars(self) -> None:
        p = SimplePortfolio()
        p.apply_fill(_buy("AAPL", qty=5, price=100.0))
        p.apply_fill(_buy("MSFT", qty=2, price=200.0))
        p.update_market([_bar("AAPL", close=110.0), _bar("MSFT", close=210.0)])
        positions = p.get_positions()
        assert positions["AAPL"].unrealized_pnl == pytest.approx(50.0)
        assert positions["MSFT"].unrealized_pnl == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# record_equity / get_equity_curve
# ---------------------------------------------------------------------------


class TestEquityCurve:
    def test_record_equity_appends_snapshot(self) -> None:
        p = SimplePortfolio(10_000.0)
        p.record_equity(_TS)
        curve = p.get_equity_curve()
        assert len(curve) == 1
        assert float(curve.iloc[0]) == pytest.approx(10_000.0)

    def test_equity_curve_indexed_by_timestamp(self) -> None:
        p = SimplePortfolio(10_000.0)
        p.record_equity(_TS)
        curve = p.get_equity_curve()
        assert curve.index[0] == pd.Timestamp(_TS)

    def test_equity_curve_grows_after_buy(self) -> None:
        p = SimplePortfolio(10_000.0)
        p.record_equity(_TS)
        p.apply_fill(_buy("AAPL", qty=10, price=100.0))
        p.update_market([_bar("AAPL", close=120.0)])
        p.record_equity(_TS2)
        curve = p.get_equity_curve()
        assert len(curve) == 2
        assert float(curve.iloc[1]) > float(curve.iloc[0])

    def test_multiple_equity_snapshots_in_order(self) -> None:
        p = SimplePortfolio(5_000.0)
        ts_list = [
            datetime(2024, 1, i, tzinfo=timezone.utc) for i in range(2, 6)
        ]
        for ts in ts_list:
            p.record_equity(ts)
        curve = p.get_equity_curve()
        assert len(curve) == 4


# ---------------------------------------------------------------------------
# get_total_equity
# ---------------------------------------------------------------------------


class TestTotalEquity:
    def test_total_equity_no_positions(self) -> None:
        p = SimplePortfolio(10_000.0)
        assert p.get_total_equity() == pytest.approx(10_000.0)

    def test_total_equity_with_open_position_uses_last_price(self) -> None:
        p = SimplePortfolio(10_000.0)
        p.apply_fill(_buy("AAPL", qty=10, price=100.0))
        p.update_market([_bar("AAPL", close=150.0)])
        expected = p.get_cash() + 10 * 150.0
        assert p.get_total_equity() == pytest.approx(expected)

    def test_total_equity_falls_after_bad_trade(self) -> None:
        p = SimplePortfolio(10_000.0)
        p.apply_fill(_buy("AAPL", qty=10, price=100.0))
        p.update_market([_bar("AAPL", close=50.0)])
        assert p.get_total_equity() < 10_000.0
