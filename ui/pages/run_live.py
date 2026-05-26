"""
Live Backtest Animation page.

Fetches real OHLCV data, builds a single Plotly animated figure (candlesticks +
equity curve using Plotly's native frames mechanism), computes SMA-20 crossover
results, saves them to the DB, and shows a tearsheet navigation button.

All animation happens client-side in the browser's JavaScript engine — there are
NO Streamlit re-renders after the initial chart render, so the page never flashes.

Import discipline
-----------------
* Imports from ``ui.auth``, ``ui.db``, and ``ui.components`` only.
* ``yfinance`` is an external data library — not the project's backtester engine.
* No backtester engine imports.
* No cross-page imports.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

from ui.auth import get_current_user, require_role
from ui.db import get_db, get_run, update_run_status

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

_SMA_WINDOW:    int = 20   # SMA period for crossover signal
_TARGET_FRAMES: int = 120  # target animation frame count (step is derived)
_FRAME_MS:      int = 35   # milliseconds per animation frame (~28 fps)
_MIN_BARS:      int = 15   # bars shown before animation starts

# ---------------------------------------------------------------------------
# Pure-computation helpers (no Streamlit)
# ---------------------------------------------------------------------------


def _fetch_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close"]].dropna()


def _sma_crossover(
    close: pd.Series,
    window: int = _SMA_WINDOW,
) -> tuple[pd.Series, pd.Series]:
    """Return (signal, sma). signal=1 means in-market (lagged by 1 bar)."""
    sma    = close.rolling(window).mean()
    signal = (close > sma).astype(float).shift(1).fillna(0.0)
    return signal, sma


def _compute_equity(
    close: pd.Series,
    signal: pd.Series,
    initial_cash: float,
) -> pd.Series:
    daily_ret      = close.pct_change().fillna(0.0)
    strat_ret      = daily_ret * signal
    equity         = initial_cash * (1.0 + strat_ret).cumprod()
    equity.iloc[0] = initial_cash
    return equity


def _build_metrics(
    equity: pd.Series,
    signal: pd.Series,
    initial_cash: float,
) -> dict:
    total_return = (equity.iloc[-1] - initial_cash) / initial_cash
    n_years      = max(len(equity) / 252.0, 0.01)
    cagr         = (equity.iloc[-1] / initial_cash) ** (1.0 / n_years) - 1.0

    strat_ret  = equity.pct_change().fillna(0.0)
    rf_daily   = 0.02 / 252.0
    excess     = strat_ret - rf_daily
    sharpe     = (excess.mean() / excess.std()) * (252.0 ** 0.5) if excess.std() > 0 else 0.0

    rolling_max = equity.cummax()
    max_dd      = float(((equity - rolling_max) / rolling_max).min())

    active      = strat_ret[strat_ret != 0.0]
    win_rate    = float((active > 0).mean()) if len(active) > 0 else 0.5

    return {
        "total_return": round(float(total_return), 4),
        "cagr":         round(float(cagr), 4),
        "sharpe_ratio": round(float(sharpe), 3),
        "max_drawdown": round(float(max_dd), 4),
        "win_rate":     round(float(win_rate), 4),
        "total_trades": int(signal.diff().abs().sum()),
    }


# ---------------------------------------------------------------------------
# Chart builder — one animated figure, rendered once
# ---------------------------------------------------------------------------


def _build_animated_figure(
    df: pd.DataFrame,
    sma: pd.Series,
    equity_curve: pd.Series,
    ticker: str,
    initial_cash: float,
) -> go.Figure:
    """
    Build a two-row Plotly figure with animation frames.

    Row 1 — candlestick chart + SMA line (builds left-to-right).
    Row 2 — equity curve (grows as bars are added).

    The figure is rendered once by Streamlit; all animation happens in the
    browser's JS engine with no further Python involvement.
    """
    n    = len(df)
    step = max(1, n // _TARGET_FRAMES)

    # Pre-compute y-axis ranges from full data so axes don't jump mid-animation
    y_lo  = float(df["Low"].min())  * 0.97
    y_hi  = float(df["High"].max()) * 1.03
    eq_lo = float(equity_curve.min()) * 0.97
    eq_hi = float(equity_curve.max()) * 1.03
    x_lo  = df.index[0]
    x_hi  = df.index[-1]

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.62, 0.38],
        shared_xaxes=False,
        vertical_spacing=0.06,
    )

    # ── Initial state: first _MIN_BARS bars ────────────────────────────────
    init_df  = df.iloc[:_MIN_BARS]
    init_sma = sma.reindex(init_df.index)
    init_eq  = equity_curve.iloc[:_MIN_BARS]

    # Trace 0 — candlestick
    fig.add_trace(
        go.Candlestick(
            x=init_df.index,
            open=init_df["Open"], high=init_df["High"],
            low=init_df["Low"],   close=init_df["Close"],
            name=ticker,
            increasing_line_color="#3a7d1e",
            decreasing_line_color="#c0392b",
            increasing_fillcolor="rgba(58,125,30,0.75)",
            decreasing_fillcolor="rgba(192,57,43,0.75)",
            whiskerwidth=0.4,
        ),
        row=1, col=1,
    )

    # Trace 1 — SMA line
    fig.add_trace(
        go.Scatter(
            x=init_sma.index, y=init_sma.values,
            mode="lines",
            name=f"SMA {_SMA_WINDOW}",
            line=dict(color="#4f6ef7", width=1.5, dash="dot"),
        ),
        row=1, col=1,
    )

    # Trace 2 — reference line at initial_cash (static, not in frames)
    fig.add_trace(
        go.Scatter(
            x=[x_lo, x_hi], y=[initial_cash, initial_cash],
            mode="lines",
            line=dict(color="rgba(128,128,128,0.35)", width=1, dash="dash"),
            showlegend=False,
        ),
        row=2, col=1,
    )

    # Trace 3 — equity curve
    fig.add_trace(
        go.Scatter(
            x=init_eq.index, y=init_eq.values,
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(79,110,247,0.08)",
            line=dict(color="#4f6ef7", width=2),
            name="Portfolio",
        ),
        row=2, col=1,
    )

    # ── Animation frames ────────────────────────────────────────────────────
    frame_ends = list(range(_MIN_BARS, n, step)) + [n]
    frames = []
    for i in frame_ends:
        chunk    = df.iloc[:i]
        sma_c    = sma.reindex(chunk.index)
        eq_c     = equity_curve.iloc[:i]

        frames.append(go.Frame(
            data=[
                go.Candlestick(
                    x=chunk.index,
                    open=chunk["Open"], high=chunk["High"],
                    low=chunk["Low"],   close=chunk["Close"],
                    increasing_line_color="#3a7d1e",
                    decreasing_line_color="#c0392b",
                    increasing_fillcolor="rgba(58,125,30,0.75)",
                    decreasing_fillcolor="rgba(192,57,43,0.75)",
                    whiskerwidth=0.4,
                ),
                go.Scatter(
                    x=sma_c.index, y=sma_c.values,
                    mode="lines",
                    line=dict(color="#4f6ef7", width=1.5, dash="dot"),
                ),
                go.Scatter(
                    x=eq_c.index, y=eq_c.values,
                    mode="lines",
                    fill="tozeroy",
                    fillcolor="rgba(79,110,247,0.08)",
                    line=dict(color="#4f6ef7", width=2),
                ),
            ],
            traces=[0, 1, 3],  # do NOT update trace 2 (static reference line)
            name=str(i),
        ))

    fig.frames = frames

    # ── Layout ──────────────────────────────────────────────────────────────
    fig.update_layout(
        height=540,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=90),
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h", x=0, y=1.04,
            font_size=11, bgcolor="rgba(0,0,0,0)",
        ),

        # Play / Pause buttons
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=-0.14, x=0.0, xanchor="left",
            direction="left",
            pad={"r": 8},
            bgcolor="rgba(79,110,247,0.12)",
            bordercolor="rgba(79,110,247,0.4)",
            font=dict(size=13),
            buttons=[
                dict(
                    label="▶  Play",
                    method="animate",
                    args=[None, {
                        "frame":       {"duration": _FRAME_MS, "redraw": True},
                        "fromcurrent": True,
                        "transition":  {"duration": 0},
                    }],
                ),
                dict(
                    label="⏸  Pause",
                    method="animate",
                    args=[[None], {
                        "frame":      {"duration": 0, "redraw": False},
                        "mode":       "immediate",
                        "transition": {"duration": 0},
                    }],
                ),
            ],
        )],

        # Scrub slider
        sliders=[dict(
            active=0,
            steps=[
                dict(
                    method="animate",
                    args=[[f.name], {
                        "mode":       "immediate",
                        "frame":      {"duration": _FRAME_MS, "redraw": True},
                        "transition": {"duration": 0},
                    }],
                    label="",
                )
                for f in frames
            ],
            y=-0.08, x=0.18, len=0.8,
            pad={"t": 8},
            currentvalue=dict(
                prefix=f"{ticker} · bars processed: ",
                visible=True,
                xanchor="left",
                font=dict(size=11, color="rgba(150,150,150,0.9)"),
            ),
            transition={"duration": 0},
        )],
    )

    # Fix axis ranges so they stay stable while frames play
    fig.update_xaxes(range=[x_lo, x_hi], row=1, col=1)
    fig.update_xaxes(range=[x_lo, x_hi], row=2, col=1)
    fig.update_yaxes(range=[y_lo, y_hi], row=1, col=1,
                     tickprefix="$", tickformat=",.0f",
                     gridcolor="rgba(128,128,128,0.15)")
    fig.update_yaxes(range=[eq_lo, eq_hi], row=2, col=1,
                     tickprefix="$", tickformat=",.0f",
                     gridcolor="rgba(128,128,128,0.1)")
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.15)", row=1, col=1)
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.1)", row=2, col=1)

    return fig


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_live_page() -> None:
    """
    Render the live backtest animation page.

    Flow
    ----
    1. Guard: no run_id → warning; run not found → error.
    2. If run is already finished, skip to ``run_detail`` immediately.
    3. Fetch OHLCV data (spinner).
    4. Compute SMA-20 crossover equity curve and metrics.
    5. Persist results to DB (status → ``"DONE"``) so tearsheet has data.
    6. Render the animated Plotly figure **once** — all animation is client-side.
    7. Show final metrics strip and "View Tearsheet" navigation button.
    """
    require_role(["admin", "analyst"])
    user = get_current_user()
    assert user is not None

    run_id: str | None = st.session_state.get("selected_run_id")
    if not run_id:
        st.warning("No run selected. Go to **Run History** to pick a run.")
        return

    with get_db() as db:
        run = get_run(db, run_id)
        if run is None:
            st.error(f"Run not found: {run_id!r}")
            return

    if run.status != "RUNNING":
        st.session_state["page"] = "run_detail"
        st.rerun()
        return

    params       = run.params or {}
    symbols: list = params.get("symbols", ["SPY"])
    ticker       = (symbols[0] if symbols else "SPY").upper()
    start        = params.get("start_date",    "2020-01-01")
    end          = params.get("end_date",      "2023-12-31")
    initial_cash = float(params.get("initial_capital", 100_000))

    # ── Header ─────────────────────────────────────────────────────────────
    st.title(f"⚡ {run.strategy_name}")
    st.caption(
        f"SMA-{_SMA_WINDOW} crossover · **{ticker}** · "
        f"{start} → {end} · ${initial_cash:,.0f} starting capital"
    )

    # ── Fetch data ──────────────────────────────────────────────────────────
    with st.spinner(f"Fetching {ticker} price history…"):
        try:
            df = _fetch_ohlcv(ticker, start, end)
        except Exception as exc:
            st.error(f"Market data fetch failed: {exc}")
            with get_db() as db:
                update_run_status(db, run_id, "FAILED", error_message=str(exc))
            return

    if df.empty:
        st.error(f"No price data for **{ticker}** in {start}–{end}.")
        with get_db() as db:
            update_run_status(db, run_id, "FAILED", error_message="empty data")
        return

    # ── Compute ─────────────────────────────────────────────────────────────
    signal, sma  = _sma_crossover(df["Close"])
    equity_curve = _compute_equity(df["Close"], signal, initial_cash)
    metrics      = _build_metrics(equity_curve, signal, initial_cash)
    equity_dict  = {
        d.strftime("%Y-%m-%d"): round(float(v), 2)
        for d, v in equity_curve.items()
    }

    # ── Persist results NOW so tearsheet works on any navigation ────────────
    with get_db() as db:
        update_run_status(
            db, run_id, "DONE",
            result={"metrics": metrics, "equity_curve": equity_dict},
        )

    # ── Animated chart — rendered once, plays in browser ────────────────────
    st.markdown("Press **▶ Play** to watch the backtest run bar-by-bar.")
    fig = _build_animated_figure(df, sma, equity_curve, ticker, initial_cash)
    st.plotly_chart(fig, use_container_width=True, key="live_chart")

    # ── Final metrics strip ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Results")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Sharpe",  f"{metrics['sharpe_ratio']:.2f}")
    with m2:
        cagr_pct = metrics["cagr"] * 100
        st.metric("CAGR",    f"{cagr_pct:+.1f}%")
    with m3:
        dd_pct = metrics["max_drawdown"] * 100
        st.metric("Max DD",  f"{dd_pct:.1f}%")
    with m4:
        wr_pct = metrics["win_rate"] * 100
        st.metric("Win Rate", f"{wr_pct:.1f}%")

    st.markdown(
        f"**{metrics['total_trades']}** round-trip trades · "
        f"total return **{metrics['total_return'] * 100:+.1f}%**"
    )

    # ── Navigation ──────────────────────────────────────────────────────────
    st.markdown("")
    if st.button("View Full Tearsheet →", type="primary", key="rl_to_detail"):
        st.session_state["page"] = "run_detail"
        st.rerun()
