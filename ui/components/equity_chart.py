"""
Reusable Plotly equity-curve chart component.

Exposes a single public function :func:`equity_chart` that renders a
Plotly line chart via ``st.plotly_chart``.

Import discipline
-----------------
* Imports ``streamlit``, ``plotly``, and ``pandas`` only.
* No ``ui.*`` imports.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Palette for additional compare curves (cycles if more than 4)
_COMPARE_COLORS = ["#0C447C", "#27500A", "#791F1F", "#7A4F00"]

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _to_lists(
    curve: dict | pd.Series,
) -> tuple[list, list[float]]:
    """
    Normalise *curve* to a pair of (x_values, y_values) lists.

    Parameters
    ----------
    curve:
        Either a ``dict`` mapping timestamps/labels to portfolio values, or a
        ``pd.Series`` with the index as the x-axis.

    Returns
    -------
    tuple[list, list[float]]
        ``(x_list, y_list)`` ready for Plotly traces.
    """
    if isinstance(curve, dict):
        return list(curve.keys()), [float(v) for v in curve.values()]
    # pd.Series
    return curve.index.tolist(), [float(v) for v in curve.tolist()]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def equity_chart(
    equity_curve: dict | pd.Series,
    title: str = "Equity Curve",
    height: int = 300,
    compare_curves: list[tuple[str, dict | pd.Series]] | None = None,
) -> None:
    """
    Render a Plotly equity curve via ``st.plotly_chart``.

    Features
    --------
    * Primary curve: dark line with a low-opacity filled area beneath it.
    * Optional *compare_curves*: each extra series is overlaid in a distinct
      colour (no fill).
    * Dashed grey horizontal reference line at the initial portfolio value.
    * Clean fintech style: white background, light grey gridlines.
    * Hover shows date + value formatted as ``$1,234``.
    * Always rendered full-width (``use_container_width=True``).

    Parameters
    ----------
    equity_curve:
        Primary equity curve.  Either a ``dict`` keyed by date/timestamp, or
        a ``pd.Series`` with a datetime index.
    title:
        Chart title shown above the plot.
    height:
        Chart height in pixels (default 300).
    compare_curves:
        Additional curves to overlay.  Each element is a
        ``(label, curve)`` tuple where *curve* follows the same format as
        *equity_curve*.
    """
    xs, ys = _to_lists(equity_curve)

    fig = go.Figure()

    # ── Primary trace ──────────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            name=title,
            line=dict(color="#1a1a1a", width=2),
            fill="tozeroy",
            fillcolor="rgba(26,26,26,0.05)",
            hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>",
        )
    )

    # ── Initial value reference line ───────────────────────────────────────
    if ys:
        fig.add_hline(
            y=ys[0],
            line_dash="dash",
            line_color="#cccccc",
            line_width=1,
            annotation_text=f"Initial: ${ys[0]:,.0f}",
            annotation_position="top right",
            annotation_font_color="#aaaaaa",
        )

    # ── Compare curves ─────────────────────────────────────────────────────
    for i, (label, curve) in enumerate(compare_curves or []):
        cxs, cys = _to_lists(curve)
        color = _COMPARE_COLORS[i % len(_COMPARE_COLORS)]
        fig.add_trace(
            go.Scatter(
                x=cxs,
                y=cys,
                mode="lines",
                name=label,
                line=dict(color=color, width=2),
                hovertemplate=f"{label}<br>%{{x}}<br>${{y:,.0f}}<extra></extra>",
            )
        )

    # ── Layout ─────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#1a1a1a")),
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=0, r=0, t=40, b=0),
        hovermode="x unified",
        showlegend=bool(compare_curves),
        xaxis=dict(
            gridcolor="#f0f0f0",
            showgrid=True,
            zeroline=False,
        ),
        yaxis=dict(
            tickprefix="$",
            tickformat=",.0f",
            gridcolor="#f0f0f0",
            showgrid=True,
            zeroline=False,
        ),
    )

    st.plotly_chart(fig, use_container_width=True)
