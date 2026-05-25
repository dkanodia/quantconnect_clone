"""
Pure metric computation functions for backtesting results.

No class state, no side effects — every function is a pure transformation
from input data to a scalar or Series.  All edge-case inputs (empty equity
curve, no trades, zero standard deviation) return ``0.0`` (or an empty
Series where appropriate) so callers always get a valid value.

Baseline convention
-------------------
``total_return``, ``cagr``, and ``calmar_ratio`` derive the initial portfolio
value from ``equity.iloc[0]``.  In the standard Backtester workflow the first
bar records equity *before* any orders have been executed, so
``equity.iloc[0] == initial_cash``.

``pnl`` column fallback
-----------------------
All trade-metric functions accept the trades ``DataFrame`` produced by
``SimplePortfolio.get_trades()``, which always contains a ``pnl`` column.
If the column is absent (e.g. a custom reporter), PnL is computed as::

    (exit_price − entry_price) × quantity − commission

This fallback assumes the presence of ``exit_price``, ``entry_price``,
``quantity``, and (optionally) ``commission`` columns.

Metric summary
--------------
total_return      : (final − initial) / initial
cagr              : annualised compound growth rate
sharpe_ratio      : mean excess return / std × √periods_per_year
sortino_ratio     : mean excess return / downside_std × √periods_per_year
max_drawdown      : minimum of drawdown_series  (≤ 0)
drawdown_series   : per-bar (equity − running_peak) / running_peak  (≤ 0)
calmar_ratio      : cagr / |max_drawdown|
win_rate          : fraction of trades with pnl > 0
profit_factor     : Σ(wins) / |Σ(losses)|
avg_trade_return  : mean pnl across all trades
avg_win           : mean pnl of winning trades
avg_loss          : mean pnl of losing trades (negative value)
num_trades        : total completed trades
compute_all       : aggregate dict of every metric above
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from backtester.interfaces import BacktestResult


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _pnl_series(trades: pd.DataFrame) -> pd.Series:
    """
    Return the PnL ``Series`` for *trades*, computing it if absent.

    Preferred: ``trades["pnl"]`` column (always present from SimplePortfolio).
    Fallback:  ``(exit_price − entry_price) × quantity − commission``.
    """
    if len(trades) == 0:
        return pd.Series(dtype=float)
    if "pnl" in trades.columns:
        return trades["pnl"].astype(float)
    commission: float | pd.Series = (
        trades["commission"].astype(float)
        if "commission" in trades.columns
        else 0.0
    )
    return (
        (trades["exit_price"] - trades["entry_price"]).astype(float)
        * trades["quantity"].astype(float)
        - commission
    )


# ---------------------------------------------------------------------------
# Equity-curve metrics
# ---------------------------------------------------------------------------


def total_return(equity: pd.Series) -> float:
    """
    Total percentage return over the backtest period.

    Uses ``equity.iloc[0]`` as the starting baseline.

    Parameters
    ----------
    equity : pd.Series
        Time-indexed series of total portfolio equity values.

    Returns
    -------
    float
        ``(final − initial) / initial``, or ``0.0`` if the series has fewer
        than 2 points or the initial value is zero.
    """
    if len(equity) < 2:
        return 0.0
    initial = float(equity.iloc[0])
    if initial == 0.0:
        return 0.0
    return float((equity.iloc[-1] - initial) / initial)


def cagr(equity: pd.Series, periods_per_year: int = 252) -> float:
    """
    Compound Annual Growth Rate.

    Uses ``equity.iloc[0]`` as the starting baseline and treats each bar as
    ``1 / periods_per_year`` of a year.

    Parameters
    ----------
    equity : pd.Series
        Time-indexed series of total portfolio equity values.
    periods_per_year : int
        Number of bars per year (252 for daily, 52 for weekly, 12 monthly).

    Returns
    -------
    float
        Annualised growth rate, or ``0.0`` on edge-case inputs (fewer than 2
        bars, non-positive initial / final equity, zero duration).
    """
    n = len(equity)
    if n < 2 or periods_per_year <= 0:
        return 0.0
    initial = float(equity.iloc[0])
    if initial <= 0.0:
        return 0.0
    final = float(equity.iloc[-1])
    if final <= 0.0:
        return 0.0
    years = n / periods_per_year
    return float((final / initial) ** (1.0 / years) - 1.0)


def sharpe_ratio(
    equity: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Annualised Sharpe ratio.

    ``mean(daily_returns − rf_per_period) / std(daily_returns) × √periods``

    Parameters
    ----------
    equity : pd.Series
        Time-indexed series of total portfolio equity values.
    risk_free_rate : float
        Annual risk-free rate as a decimal (e.g. ``0.04`` for 4 %).
    periods_per_year : int
        Bars per year — used to scale the rf rate and annualise the result.

    Returns
    -------
    float
        Annualised Sharpe ratio, or ``0.0`` if there are fewer than 2 bars
        or the standard deviation of returns is zero.
    """
    if len(equity) < 2 or periods_per_year <= 0:
        return 0.0
    daily_ret = equity.pct_change().dropna()
    if len(daily_ret) == 0:
        return 0.0
    rf_per_period = risk_free_rate / periods_per_year
    excess = daily_ret - rf_per_period
    std = float(excess.std(ddof=1))
    if std == 0.0:
        return 0.0
    return float(excess.mean() / std * math.sqrt(periods_per_year))


def sortino_ratio(
    equity: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Annualised Sortino ratio.

    Uses the downside deviation (root-mean-square of negative excess returns)
    as the denominator instead of total standard deviation.

    Parameters
    ----------
    equity : pd.Series
        Time-indexed series of total portfolio equity values.
    risk_free_rate : float
        Annual risk-free rate as a decimal.
    periods_per_year : int
        Bars per year.

    Returns
    -------
    float
        Annualised Sortino ratio, or ``0.0`` if there are no negative-return
        periods or fewer than 2 bars.
    """
    if len(equity) < 2 or periods_per_year <= 0:
        return 0.0
    daily_ret = equity.pct_change().dropna()
    if len(daily_ret) == 0:
        return 0.0
    rf_per_period = risk_free_rate / periods_per_year
    excess = daily_ret - rf_per_period
    downside = excess[excess < 0.0]
    if len(downside) == 0:
        return 0.0
    downside_std = math.sqrt(float((downside ** 2).mean()))
    if downside_std == 0.0:
        return 0.0
    return float(excess.mean() / downside_std * math.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    """
    Maximum drawdown — the largest peak-to-trough decline as a fraction.

    Parameters
    ----------
    equity : pd.Series
        Time-indexed series of total portfolio equity values.

    Returns
    -------
    float
        Maximum drawdown as a **negative** fraction (e.g. ``-0.23`` for a
        23 % drawdown), or ``0.0`` if the series is empty or has no decline.
    """
    if len(equity) == 0:
        return 0.0
    return float(drawdown_series(equity).min())


def drawdown_series(equity: pd.Series) -> pd.Series:
    """
    Per-bar fractional drawdown from the running equity peak.

    Parameters
    ----------
    equity : pd.Series
        Time-indexed series of total portfolio equity values.

    Returns
    -------
    pd.Series
        Series with the same index as *equity*, values ≤ 0.
        Empty if *equity* is empty.
    """
    if len(equity) == 0:
        return pd.Series(dtype=float)
    running_max = equity.cummax()
    safe_max = running_max.where(running_max != 0.0, other=float("nan"))
    return (equity - running_max).divide(safe_max).fillna(0.0)


def calmar_ratio(equity: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calmar ratio — CAGR divided by the absolute maximum drawdown.

    Returns ``0.0`` if the maximum drawdown is zero (no decline ever occurred).

    Parameters
    ----------
    equity : pd.Series
        Time-indexed series of total portfolio equity values.
    periods_per_year : int
        Bars per year — forwarded to :func:`cagr`.

    Returns
    -------
    float
        Calmar ratio.
    """
    mdd = max_drawdown(equity)
    if mdd == 0.0:
        return 0.0
    return float(cagr(equity, periods_per_year) / abs(mdd))


# ---------------------------------------------------------------------------
# Trade-based metrics
# ---------------------------------------------------------------------------


def win_rate(trades: pd.DataFrame) -> float:
    """
    Fraction of completed trades that produced a positive PnL.

    Parameters
    ----------
    trades : pd.DataFrame
        Trades DataFrame (see module docstring for ``pnl`` column rules).

    Returns
    -------
    float
        Value in ``[0, 1]``, or ``0.0`` if there are no trades.
    """
    pnl = _pnl_series(trades)
    if len(pnl) == 0:
        return 0.0
    return float((pnl > 0).mean())


def profit_factor(trades: pd.DataFrame) -> float:
    """
    Gross profit divided by gross loss.

    Parameters
    ----------
    trades : pd.DataFrame
        Trades DataFrame (see module docstring for ``pnl`` column rules).

    Returns
    -------
    float
        Profit factor (> 1 is net profitable), or ``0.0`` if there are no
        trades or no losing trades (avoids division by zero).
    """
    pnl = _pnl_series(trades)
    if len(pnl) == 0:
        return 0.0
    winners = pnl[pnl > 0]
    losers = pnl[pnl < 0]
    if len(losers) == 0:
        return 0.0
    gross_loss = float(abs(losers.sum()))
    if gross_loss == 0.0:
        return 0.0
    return float(winners.sum()) / gross_loss


def avg_trade_return(trades: pd.DataFrame) -> float:
    """
    Average PnL per completed trade.

    Parameters
    ----------
    trades : pd.DataFrame
        Trades DataFrame (see module docstring for ``pnl`` column rules).

    Returns
    -------
    float
        Mean PnL value, or ``0.0`` if there are no trades.
    """
    pnl = _pnl_series(trades)
    if len(pnl) == 0:
        return 0.0
    return float(pnl.mean())


def avg_win(trades: pd.DataFrame) -> float:
    """
    Average PnL of winning trades (PnL strictly > 0).

    Parameters
    ----------
    trades : pd.DataFrame
        Trades DataFrame (see module docstring for ``pnl`` column rules).

    Returns
    -------
    float
        Mean winning PnL, or ``0.0`` if there are no winners.
    """
    pnl = _pnl_series(trades)
    winners = pnl[pnl > 0]
    if len(winners) == 0:
        return 0.0
    return float(winners.mean())


def avg_loss(trades: pd.DataFrame) -> float:
    """
    Average PnL of losing trades (PnL strictly < 0).

    Parameters
    ----------
    trades : pd.DataFrame
        Trades DataFrame (see module docstring for ``pnl`` column rules).

    Returns
    -------
    float
        Mean losing PnL (a negative value), or ``0.0`` if there are no losers.
    """
    pnl = _pnl_series(trades)
    losers = pnl[pnl < 0]
    if len(losers) == 0:
        return 0.0
    return float(losers.mean())


def num_trades(trades: pd.DataFrame) -> int:
    """
    Total number of completed trades (rows in the trades DataFrame).

    Parameters
    ----------
    trades : pd.DataFrame
        Completed trades DataFrame.

    Returns
    -------
    int
        Row count.
    """
    return len(trades)


# ---------------------------------------------------------------------------
# Aggregate helper
# ---------------------------------------------------------------------------


def compute_all(result: BacktestResult) -> dict[str, Any]:
    """
    Compute the full analytics metric set from a ``BacktestResult``.

    All inputs are derived from ``result.equity_curve`` and
    ``result.trades``; no external parameters are required.

    Parameters
    ----------
    result : BacktestResult
        The completed backtest result, as returned by ``Backtester.run()``.

    Returns
    -------
    dict[str, Any]
        Flat dictionary mapping metric name (snake_case) → plain Python
        scalar (``float`` or ``int``).  Every value is safe to
        JSON-serialise.  Keys (in order):

        ``total_return``, ``final_equity``, ``cagr``, ``sharpe_ratio``,
        ``sortino_ratio``, ``max_drawdown``, ``calmar_ratio``, ``win_rate``,
        ``profit_factor``, ``avg_trade_return``, ``avg_win``, ``avg_loss``,
        ``num_trades``.
    """
    ec = result.equity_curve
    trd = result.trades
    final_eq = float(ec.iloc[-1]) if len(ec) > 0 else 0.0

    return {
        "total_return": total_return(ec),
        "final_equity": final_eq,
        "cagr": cagr(ec),
        "sharpe_ratio": sharpe_ratio(ec),
        "sortino_ratio": sortino_ratio(ec),
        "max_drawdown": max_drawdown(ec),
        "calmar_ratio": calmar_ratio(ec),
        "win_rate": win_rate(trd),
        "profit_factor": profit_factor(trd),
        "avg_trade_return": avg_trade_return(trd),
        "avg_win": avg_win(trd),
        "avg_loss": avg_loss(trd),
        "num_trades": num_trades(trd),
    }
