"""
PlotlyTearsheet — five-panel interactive tearsheet built with Plotly.

Panel layout (top → bottom)
-----------------------------
1. Equity curve (35 %)   — line chart with a horizontal reference at initial
                           equity.
2. Drawdown    (15 %)   — filled area chart, filled red below zero.
3. Monthly returns heatmap (20 %) — year × month pivot, green/red colourscale.
4. Trade P&L distribution (15 %) — histogram of per-trade PnL.
5. Metrics table (15 %)  — ``go.Table`` with all metrics from
                            :func:`~backtester.analytics.metrics.compute_all`,
                            values formatted to convention.

Visual style: ``plotly_dark`` template, 900 px tall.

Usage
-----
>>> reporter = PlotlyTearsheet()
>>> fig = reporter.report(result)   # returns Figure; does NOT call .show()
>>> fig.write_html("tearsheet.html")
"""

from __future__ import annotations

import calendar
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backtester.analytics.metrics import compute_all, drawdown_series
from backtester.interfaces import BacktestResult, Reporter

_ROW_HEIGHTS = [0.35, 0.15, 0.20, 0.15, 0.15]
_TEMPLATE = "plotly_dark"
_HEIGHT = 900

# Metric formatting rules: key → (format_string, label)
# %  → render as percentage    .2f → 2 decimal places    d → integer
_METRIC_FMT: dict[str, tuple[str, str]] = {
    "total_return":    ("{:.2%}", "Total Return"),
    "cagr":            ("{:.2%}", "CAGR"),
    "sharpe_ratio":    ("{:.2f}", "Sharpe"),
    "sortino_ratio":   ("{:.2f}", "Sortino"),
    "max_drawdown":    ("{:.2%}", "Max Drawdown"),
    "calmar_ratio":    ("{:.2f}", "Calmar"),
    "win_rate":        ("{:.2%}", "Win Rate"),
    "profit_factor":   ("{:.2f}", "Profit Factor"),
    "avg_trade_return":("{:,.2f}", "Avg Trade PnL"),
    "avg_win":         ("{:,.2f}", "Avg Win"),
    "avg_loss":        ("{:,.2f}", "Avg Loss"),
    "num_trades":      ("{:d}", "# Trades"),
}


def _fmt_metric(key: str, value: Any) -> str:
    """Return a human-readable string for a single metric value."""
    if key not in _METRIC_FMT:
        return str(value)
    fmt, _ = _METRIC_FMT[key]
    try:
        if fmt == "{:d}":
            return fmt.format(int(value))
        return fmt.format(float(value))
    except (TypeError, ValueError):
        return str(value)


class PlotlyTearsheet(Reporter):
    """
    Generate a five-panel Plotly tearsheet from a ``BacktestResult``.

    Implements the :class:`~backtester.interfaces.Reporter` interface.
    Call :meth:`report` to produce a ``plotly.graph_objects.Figure``.

    The figure is **not** displayed automatically — call ``fig.show()`` or
    ``fig.write_html(path)`` on the returned object.

    Parameters
    ----------
    height : int
        Overall figure height in pixels.  Defaults to 900.
    template : str
        Plotly template name.  Defaults to ``"plotly_dark"``.
    """

    def __init__(
        self,
        height: int = _HEIGHT,
        template: str = _TEMPLATE,
    ) -> None:
        self._height = height
        self._template = template

    # ------------------------------------------------------------------
    # Reporter interface
    # ------------------------------------------------------------------

    def report(self, result: BacktestResult) -> go.Figure:
        """
        Build and return a five-panel tearsheet ``Figure``.

        Parameters
        ----------
        result : BacktestResult
            Completed backtest result.  Uses ``equity_curve``, ``trades``,
            ``metrics``, ``strategy_name``, and ``run_id``.

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive multi-panel figure — not shown automatically.
        """
        ec = result.equity_curve
        trades = result.trades
        strategy = result.strategy_name or "Strategy"
        run_id = result.run_id
        metrics = compute_all(result)

        fig = make_subplots(
            rows=5,
            cols=1,
            shared_xaxes=False,
            row_heights=_ROW_HEIGHTS,
            vertical_spacing=0.05,
            specs=[
                [{"type": "xy"}],
                [{"type": "xy"}],
                [{"type": "xy"}],
                [{"type": "xy"}],
                [{"type": "table"}],
            ],
            subplot_titles=[
                "Equity Curve",
                "Drawdown",
                "Monthly Returns (%)",
                "Trade P&L Distribution",
                "Summary Metrics",
            ],
        )

        self._add_equity_curve(fig, ec, row=1)
        self._add_drawdown(fig, ec, row=2)
        self._add_monthly_heatmap(fig, ec, row=3)
        self._add_pnl_histogram(fig, trades, row=4)
        self._add_metrics_table(fig, metrics, row=5)

        fig.update_layout(
            template=self._template,
            height=self._height,
            showlegend=False,
            margin={"l": 60, "r": 30, "t": 80, "b": 40},
            title={
                "text": (
                    f"{strategy} — Backtest Tearsheet ({run_id})"
                ),
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 15},
            },
        )

        # Axis labels
        fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        fig.update_yaxes(title_text="Return (%)", row=3, col=1)
        fig.update_xaxes(title_text="P&L ($)", row=4, col=1)
        fig.update_yaxes(title_text="Count", row=4, col=1)

        return fig

    # ------------------------------------------------------------------
    # Panel builders
    # ------------------------------------------------------------------

    def _add_equity_curve(
        self, fig: go.Figure, ec: pd.Series, *, row: int
    ) -> None:
        """Row 1 — equity line with a horizontal reference at initial equity."""
        if len(ec) == 0:
            return
        initial = float(ec.iloc[0])
        fig.add_trace(
            go.Scatter(
                x=list(ec.index),
                y=list(ec.values),
                mode="lines",
                name="Equity",
                line={"color": "#00b4d8", "width": 1.8},
                hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.2f}<extra></extra>",
            ),
            row=row,
            col=1,
        )
        # Horizontal reference line at initial equity
        fig.add_hline(
            y=initial,
            line_dash="dash",
            line_color="rgba(255,255,255,0.35)",
            row=row,
            col=1,
        )

    def _add_drawdown(
        self, fig: go.Figure, ec: pd.Series, *, row: int
    ) -> None:
        """Row 2 — drawdown as a filled area chart (red below zero)."""
        if len(ec) < 2:
            return
        dd = drawdown_series(ec)
        fig.add_trace(
            go.Scatter(
                x=list(dd.index),
                y=[v * 100 for v in dd.values],
                mode="lines",
                fill="tozeroy",
                name="Drawdown",
                line={"color": "#e63946", "width": 1.0},
                fillcolor="rgba(230,57,70,0.30)",
                hovertemplate="%{x|%Y-%m-%d}<br>DD: %{y:.2f}%<extra></extra>",
            ),
            row=row,
            col=1,
        )

    def _add_monthly_heatmap(
        self, fig: go.Figure, ec: pd.Series, *, row: int
    ) -> None:
        """Row 3 — monthly returns heatmap (year × month)."""
        if len(ec) < 2:
            return
        if not isinstance(ec.index, pd.DatetimeIndex):
            return

        # Resample to month-end equity (compatible with pandas 2.1 and 2.2+)
        try:
            monthly_ec = ec.resample("ME").last()
        except ValueError:
            monthly_ec = ec.resample("M").last()

        monthly_ret = monthly_ec.pct_change().dropna()
        if len(monthly_ret) == 0:
            return

        df = pd.DataFrame({
            "return": monthly_ret.values * 100,
            "year": monthly_ret.index.year,
            "month": monthly_ret.index.month,
        })
        try:
            pivot = df.pivot_table(
                values="return", index="year", columns="month", aggfunc="mean"
            )
        except Exception:
            return

        # Reindex columns to all 12 months so the heatmap is always full-width
        all_months = list(range(1, 13))
        pivot = pivot.reindex(columns=all_months)

        month_labels = [calendar.month_abbr[m] for m in all_months]
        year_labels = [str(y) for y in pivot.index]

        fig.add_trace(
            go.Heatmap(
                z=pivot.values.tolist(),
                x=month_labels,
                y=year_labels,
                colorscale=[
                    [0.0, "#e63946"],
                    [0.5, "#2d2d2d"],
                    [1.0, "#06d6a0"],
                ],
                zmid=0,
                showscale=True,
                hoverongaps=False,
                hovertemplate=(
                    "Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>"
                ),
            ),
            row=row,
            col=1,
        )

    def _add_pnl_histogram(
        self, fig: go.Figure, trades: pd.DataFrame, *, row: int
    ) -> None:
        """Row 4 — histogram of per-trade PnL."""
        if len(trades) == 0 or "pnl" not in trades.columns:
            return
        pnl_vals = list(trades["pnl"])
        fig.add_trace(
            go.Histogram(
                x=pnl_vals,
                nbinsx=max(10, len(pnl_vals) // 5),
                name="P&L",
                marker_color="#00b4d8",
                marker_line={"color": "#0077a8", "width": 0.5},
                hovertemplate="P&L: %{x:,.2f}<br>Count: %{y}<extra></extra>",
            ),
            row=row,
            col=1,
        )
        # Vertical zero line
        fig.add_vline(
            x=0,
            line_dash="dash",
            line_color="rgba(255,255,255,0.40)",
            row=row,
            col=1,
        )

    def _add_metrics_table(
        self, fig: go.Figure, metrics: dict[str, Any], *, row: int
    ) -> None:
        """Row 5 — two-column go.Table (Metric | Value)."""
        labels: list[str] = []
        values: list[str] = []

        for key, (_, label) in _METRIC_FMT.items():
            value = metrics.get(key)
            if value is None:
                continue
            labels.append(label)
            values.append(_fmt_metric(key, value))

        if not labels:
            return

        fig.add_trace(
            go.Table(
                header=dict(
                    values=["<b>Metric</b>", "<b>Value</b>"],
                    fill_color="#1e2a38",
                    line_color="#3a4a5a",
                    align=["left", "right"],
                    font=dict(color="#e0e0e0", size=12),
                    height=28,
                ),
                cells=dict(
                    values=[labels, values],
                    fill_color=["#151f2a", "#1a2535"],
                    line_color="#2d3d4d",
                    align=["left", "right"],
                    font=dict(color="#cccccc", size=11),
                    height=24,
                ),
            ),
            row=row,
            col=1,
        )
