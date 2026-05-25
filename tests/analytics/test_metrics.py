"""
Tests for backtester/analytics/metrics.py.

Covers:
- Each metric function with known, manually-verifiable inputs
- Edge cases: empty series, single-point series, zero std, all-losing trades
- drawdown_series values are always ≤ 0
- max_drawdown matches a manual calculation on a simple series
- compute_all returns all expected keys with float/int types
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from backtester.analytics.metrics import (
    avg_loss,
    avg_trade_return,
    avg_win,
    cagr,
    calmar_ratio,
    compute_all,
    drawdown_series,
    max_drawdown,
    num_trades,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    total_return,
    win_rate,
)
from backtester.interfaces import BacktestResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_UTC = timezone.utc

_COMPUTE_ALL_KEYS = {
    "total_return",
    "final_equity",
    "cagr",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "calmar_ratio",
    "win_rate",
    "profit_factor",
    "avg_trade_return",
    "avg_win",
    "avg_loss",
    "num_trades",
}


def _equity(values: list[float]) -> pd.Series:
    """Build a UTC-indexed daily equity Series."""
    idx = pd.date_range("2024-01-01", periods=len(values), freq="D", tz=_UTC)
    return pd.Series(values, index=idx, name="equity")


def _trades(pnls: list[float]) -> pd.DataFrame:
    """Build a trades DataFrame with the given PnL values."""
    n = len(pnls)
    if n == 0:
        return pd.DataFrame(
            columns=[
                "symbol", "exit_timestamp", "quantity",
                "entry_price", "exit_price", "pnl", "commission",
            ]
        )
    idx = pd.date_range("2024-01-02", periods=n, freq="D", tz=_UTC)
    return pd.DataFrame({
        "symbol": ["AAPL"] * n,
        "exit_timestamp": list(idx),
        "quantity": [10.0] * n,
        "entry_price": [100.0] * n,
        "exit_price": [100.0 + p / 10.0 for p in pnls],
        "pnl": pnls,
        "commission": [0.0] * n,
    })


def _result(equity_vals: list[float], pnls: list[float]) -> BacktestResult:
    return BacktestResult(
        equity_curve=_equity(equity_vals),
        trades=_trades(pnls),
        metrics={},
        params={},
        strategy_name="Test",
    )


# ---------------------------------------------------------------------------
# total_return
# ---------------------------------------------------------------------------


class TestTotalReturn:
    def test_positive(self) -> None:
        # 100k → 110k is +10%
        assert total_return(_equity([100_000, 110_000])) == pytest.approx(0.10)

    def test_negative(self) -> None:
        assert total_return(_equity([100_000, 90_000])) == pytest.approx(-0.10)

    def test_flat(self) -> None:
        assert total_return(_equity([100_000, 100_000])) == pytest.approx(0.0)

    def test_empty_returns_zero(self) -> None:
        assert total_return(pd.Series(dtype=float)) == 0.0

    def test_single_point_returns_zero(self) -> None:
        assert total_return(_equity([100_000])) == 0.0

    def test_zero_initial_returns_zero(self) -> None:
        # Degenerate: starting equity is 0
        assert total_return(_equity([0.0, 10_000])) == 0.0

    def test_multi_bar_uses_first_and_last(self) -> None:
        # Middle values are irrelevant for total return
        ec = _equity([100_000, 90_000, 95_000, 106_000])
        assert total_return(ec) == pytest.approx(0.06)

    def test_returns_float(self) -> None:
        assert isinstance(total_return(_equity([100_000, 105_000])), float)


# ---------------------------------------------------------------------------
# cagr
# ---------------------------------------------------------------------------


class TestCagr:
    def test_exactly_one_year_10pct(self) -> None:
        # 252 daily bars: 100k → 110k in exactly 252/252 = 1 year → CAGR = 10%
        vals = [100_000.0] * 251 + [110_000.0]
        assert cagr(_equity(vals)) == pytest.approx(0.10, rel=1e-6)

    def test_empty_returns_zero(self) -> None:
        assert cagr(pd.Series(dtype=float)) == 0.0

    def test_single_point_returns_zero(self) -> None:
        assert cagr(_equity([100_000])) == 0.0

    def test_zero_initial_returns_zero(self) -> None:
        assert cagr(_equity([0.0, 110_000])) == 0.0

    def test_zero_periods_per_year_returns_zero(self) -> None:
        assert cagr(_equity([100_000, 110_000]), periods_per_year=0) == 0.0

    def test_negative_final_equity_returns_zero(self) -> None:
        assert cagr(_equity([100_000, -1_000])) == 0.0

    def test_returns_float(self) -> None:
        assert isinstance(cagr(_equity([100_000, 110_000])), float)


# ---------------------------------------------------------------------------
# sharpe_ratio
# ---------------------------------------------------------------------------


class TestSharpeRatio:
    def test_positive_for_rising_equity(self) -> None:
        ec = _equity([100_000 + i * 200 for i in range(60)])
        assert sharpe_ratio(ec) > 0.0

    def test_constant_equity_zero_std_returns_zero(self) -> None:
        # Constant returns → std = 0 → Sharpe = 0
        assert sharpe_ratio(_equity([100_000.0] * 20)) == 0.0

    def test_empty_returns_zero(self) -> None:
        assert sharpe_ratio(pd.Series(dtype=float)) == 0.0

    def test_single_bar_returns_zero(self) -> None:
        assert sharpe_ratio(_equity([100_000])) == 0.0

    def test_risk_free_rate_reduces_sharpe(self) -> None:
        ec = _equity([100_000 + i * 200 for i in range(60)])
        assert sharpe_ratio(ec, risk_free_rate=0.10) < sharpe_ratio(ec)

    def test_zero_periods_per_year_returns_zero(self) -> None:
        ec = _equity([100_000, 101_000, 102_000])
        assert sharpe_ratio(ec, periods_per_year=0) == 0.0

    def test_returns_float(self) -> None:
        ec = _equity([100_000, 101_000, 99_000, 102_000])
        assert isinstance(sharpe_ratio(ec), float)


# ---------------------------------------------------------------------------
# sortino_ratio
# ---------------------------------------------------------------------------


class TestSortinoRatio:
    def test_all_positive_returns_returns_zero(self) -> None:
        # No downside → downside deviation = 0 → Sortino = 0
        ec = _equity([100_000, 101_000, 102_000, 103_000])
        assert sortino_ratio(ec) == 0.0

    def test_mixed_returns_positive_net(self) -> None:
        ec = _equity([100_000, 102_000, 101_000, 103_000, 102_500, 105_000])
        assert sortino_ratio(ec) > 0.0

    def test_empty_returns_zero(self) -> None:
        assert sortino_ratio(pd.Series(dtype=float)) == 0.0

    def test_single_bar_returns_zero(self) -> None:
        assert sortino_ratio(_equity([100_000])) == 0.0

    def test_returns_float(self) -> None:
        ec = _equity([100_000, 99_000, 101_000])
        assert isinstance(sortino_ratio(ec), float)

    def test_zero_periods_per_year_returns_zero(self) -> None:
        ec = _equity([100_000, 99_000, 101_000])
        assert sortino_ratio(ec, periods_per_year=0) == 0.0


# ---------------------------------------------------------------------------
# max_drawdown
# ---------------------------------------------------------------------------


class TestMaxDrawdown:
    def test_known_drawdown(self) -> None:
        # Peak 110k, trough 88k → DD = (88k - 110k) / 110k = -0.2
        ec = _equity([100_000, 110_000, 88_000])
        assert max_drawdown(ec) == pytest.approx(-22_000 / 110_000, rel=1e-6)

    def test_monotone_rising_is_zero(self) -> None:
        assert max_drawdown(_equity([100_000, 101_000, 102_000])) == 0.0

    def test_empty_is_zero(self) -> None:
        assert max_drawdown(pd.Series(dtype=float)) == 0.0

    def test_single_point_is_zero(self) -> None:
        assert max_drawdown(_equity([100_000])) == 0.0

    def test_returns_non_positive(self) -> None:
        ec = _equity([100_000, 110_000, 80_000, 95_000])
        assert max_drawdown(ec) <= 0.0

    def test_returns_float(self) -> None:
        assert isinstance(max_drawdown(_equity([100_000, 90_000])), float)


# ---------------------------------------------------------------------------
# drawdown_series
# ---------------------------------------------------------------------------


class TestDrawdownSeries:
    def test_all_values_non_positive(self) -> None:
        ec = _equity([100_000, 110_000, 90_000, 95_000, 105_000])
        assert (drawdown_series(ec) <= 0.0).all()

    def test_at_new_peak_value_is_zero(self) -> None:
        ec = _equity([100_000, 110_000, 90_000])
        dd = drawdown_series(ec)
        assert dd.iloc[1] == pytest.approx(0.0)

    def test_empty_input_returns_empty(self) -> None:
        dd = drawdown_series(pd.Series(dtype=float))
        assert len(dd) == 0

    def test_monotone_rising_all_zeros(self) -> None:
        ec = _equity([100_000, 101_000, 102_000])
        assert (drawdown_series(ec) == 0.0).all()

    def test_length_matches_input(self) -> None:
        ec = _equity([100_000, 90_000, 80_000, 110_000])
        assert len(drawdown_series(ec)) == len(ec)

    def test_known_trough_value(self) -> None:
        ec = _equity([100_000, 110_000, 88_000])
        dd = drawdown_series(ec)
        # At trough: (88k - 110k) / 110k = -0.2
        assert dd.iloc[-1] == pytest.approx(-22_000 / 110_000, rel=1e-6)


# ---------------------------------------------------------------------------
# calmar_ratio
# ---------------------------------------------------------------------------


class TestCalmarRatio:
    def test_zero_drawdown_returns_zero(self) -> None:
        ec = _equity([100_000, 101_000, 102_000])
        assert calmar_ratio(ec) == 0.0

    def test_empty_returns_zero(self) -> None:
        assert calmar_ratio(pd.Series(dtype=float)) == 0.0

    def test_positive_for_profitable_with_drawdown(self) -> None:
        # Equity grows overall but has a mid-run dip
        ec = _equity([100_000, 105_000, 95_000, 110_000, 105_000, 115_000])
        result = calmar_ratio(ec)
        assert result > 0.0

    def test_returns_float(self) -> None:
        ec = _equity([100_000, 105_000, 95_000, 110_000])
        assert isinstance(calmar_ratio(ec), float)

    def test_zero_periods_per_year_returns_zero(self) -> None:
        ec = _equity([100_000, 110_000, 95_000, 115_000])
        assert calmar_ratio(ec, periods_per_year=0) == 0.0


# ---------------------------------------------------------------------------
# win_rate
# ---------------------------------------------------------------------------


class TestWinRate:
    def test_all_winners(self) -> None:
        assert win_rate(_trades([100.0, 200.0, 50.0])) == pytest.approx(1.0)

    def test_all_losers(self) -> None:
        assert win_rate(_trades([-100.0, -50.0])) == pytest.approx(0.0)

    def test_mixed(self) -> None:
        # 2 wins out of 3
        assert win_rate(_trades([100.0, -50.0, 200.0])) == pytest.approx(2.0 / 3.0)

    def test_empty_returns_zero(self) -> None:
        assert win_rate(_trades([])) == 0.0

    def test_zero_pnl_is_not_a_win(self) -> None:
        assert win_rate(_trades([0.0, 100.0])) == pytest.approx(0.5)

    def test_returns_float(self) -> None:
        assert isinstance(win_rate(_trades([10.0])), float)


# ---------------------------------------------------------------------------
# profit_factor
# ---------------------------------------------------------------------------


class TestProfitFactor:
    def test_known_value(self) -> None:
        # 300 gross profit / 50 gross loss = 6.0
        assert profit_factor(_trades([100.0, 200.0, -50.0])) == pytest.approx(6.0)

    def test_no_losses_returns_zero(self) -> None:
        assert profit_factor(_trades([100.0, 200.0])) == 0.0

    def test_empty_returns_zero(self) -> None:
        assert profit_factor(_trades([])) == 0.0

    def test_all_losses_gross_profit_zero(self) -> None:
        # winners sum = 0 → 0 / gross_loss = 0.0
        assert profit_factor(_trades([-100.0, -50.0])) == pytest.approx(0.0)

    def test_returns_float(self) -> None:
        assert isinstance(profit_factor(_trades([10.0, -5.0])), float)


# ---------------------------------------------------------------------------
# avg_trade_return
# ---------------------------------------------------------------------------


class TestAvgTradeReturn:
    def test_known_mean(self) -> None:
        expected = (100.0 - 50.0 + 200.0) / 3.0
        assert avg_trade_return(_trades([100.0, -50.0, 200.0])) == pytest.approx(expected)

    def test_empty_returns_zero(self) -> None:
        assert avg_trade_return(_trades([])) == 0.0

    def test_single_trade(self) -> None:
        assert avg_trade_return(_trades([75.0])) == pytest.approx(75.0)


# ---------------------------------------------------------------------------
# avg_win
# ---------------------------------------------------------------------------


class TestAvgWin:
    def test_known_mean(self) -> None:
        assert avg_win(_trades([100.0, 200.0, -50.0])) == pytest.approx(150.0)

    def test_no_winners_returns_zero(self) -> None:
        assert avg_win(_trades([-10.0, -20.0])) == 0.0

    def test_empty_returns_zero(self) -> None:
        assert avg_win(_trades([])) == 0.0


# ---------------------------------------------------------------------------
# avg_loss
# ---------------------------------------------------------------------------


class TestAvgLoss:
    def test_known_mean(self) -> None:
        assert avg_loss(_trades([100.0, -50.0, -100.0])) == pytest.approx(-75.0)

    def test_no_losers_returns_zero(self) -> None:
        assert avg_loss(_trades([10.0, 20.0])) == 0.0

    def test_empty_returns_zero(self) -> None:
        assert avg_loss(_trades([])) == 0.0

    def test_returns_negative(self) -> None:
        assert avg_loss(_trades([-100.0])) < 0.0


# ---------------------------------------------------------------------------
# num_trades
# ---------------------------------------------------------------------------


class TestNumTrades:
    def test_count(self) -> None:
        assert num_trades(_trades([10.0, -5.0, 20.0])) == 3

    def test_empty_is_zero(self) -> None:
        assert num_trades(_trades([])) == 0

    def test_returns_int(self) -> None:
        assert isinstance(num_trades(_trades([1.0])), int)


# ---------------------------------------------------------------------------
# compute_all
# ---------------------------------------------------------------------------


class TestComputeAll:
    def test_returns_all_expected_keys(self) -> None:
        r = _result([100_000, 101_000, 102_000, 105_000], [200.0, -50.0])
        assert _COMPUTE_ALL_KEYS.issubset(set(compute_all(r).keys()))

    def test_all_values_are_plain_python_scalars(self) -> None:
        r = _result([100_000, 101_000, 102_000], [100.0, -50.0])
        for key, val in compute_all(r).items():
            assert isinstance(val, (int, float)), (
                f"metric '{key}' has unexpected type {type(val).__name__}"
            )

    def test_total_return_correct(self) -> None:
        r = _result([100_000, 110_000], [])
        assert compute_all(r)["total_return"] == pytest.approx(0.10)

    def test_max_drawdown_non_positive(self) -> None:
        r = _result([100_000, 110_000, 90_000, 105_000], [])
        assert compute_all(r)["max_drawdown"] <= 0.0

    def test_num_trades_matches_dataframe(self) -> None:
        r = _result([100_000, 110_000], [100.0, -50.0])
        assert compute_all(r)["num_trades"] == 2

    def test_empty_equity_curve_no_raises(self) -> None:
        r = BacktestResult(
            equity_curve=pd.Series(dtype=float),
            trades=_trades([]),
            metrics={},
            params={},
            strategy_name="Empty",
        )
        m = compute_all(r)
        assert _COMPUTE_ALL_KEYS.issubset(set(m.keys()))
        assert m["total_return"] == 0.0
        assert m["num_trades"] == 0

    def test_no_trades_win_rate_and_profit_factor_zero(self) -> None:
        r = _result([100_000, 101_000, 102_000], [])
        m = compute_all(r)
        assert m["win_rate"] == 0.0
        assert m["profit_factor"] == 0.0

    def test_all_losing_trades(self) -> None:
        r = _result([100_000, 99_000, 98_000], [-100.0, -200.0])
        m = compute_all(r)
        assert m["win_rate"] == 0.0
        assert m["avg_loss"] < 0.0

    def test_pnl_fallback_without_pnl_column(self) -> None:
        """win_rate etc. still work when 'pnl' is absent (fallback path)."""
        trades_no_pnl = pd.DataFrame({
            "symbol": ["AAPL"],
            "exit_timestamp": [datetime(2024, 1, 2, tzinfo=_UTC)],
            "quantity": [10.0],
            "entry_price": [100.0],
            "exit_price": [110.0],
            "commission": [0.0],
            # no 'pnl' column
        })
        r = BacktestResult(
            equity_curve=_equity([100_000, 101_000]),
            trades=trades_no_pnl,
            metrics={},
            params={},
            strategy_name="FallbackTest",
        )
        m = compute_all(r)
        assert m["num_trades"] == 1
        assert m["win_rate"] == pytest.approx(1.0)  # (110-100)*10 = +100 > 0
