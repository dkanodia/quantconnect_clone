"""
Analytics layer — metric computation and result reporting.

Public API
----------
Metrics (pure functions):
    compute_all, total_return, cagr, sharpe_ratio, sortino_ratio,
    max_drawdown, drawdown_series, calmar_ratio, win_rate, profit_factor,
    avg_trade_return, avg_win, avg_loss, num_trades

Reporters (implement Reporter ABC):
    PlotlyTearsheet  — returns plotly.graph_objects.Figure
    DictReporter     — returns JSON-safe dict
    CSVReporter      — writes CSV files, returns output Path
"""

from backtester.analytics.csv_reporter import CSVReporter
from backtester.analytics.dict_reporter import DictReporter
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
from backtester.analytics.plotly_tearsheet import PlotlyTearsheet

__all__ = [
    # metrics
    "compute_all",
    "total_return",
    "cagr",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "drawdown_series",
    "calmar_ratio",
    "win_rate",
    "profit_factor",
    "avg_trade_return",
    "avg_win",
    "avg_loss",
    "num_trades",
    # reporters
    "PlotlyTearsheet",
    "DictReporter",
    "CSVReporter",
]
