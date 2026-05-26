"""
Reusable metric card grid component.

Exposes two public functions:

* :func:`metrics_grid` — render a full grid of metric cards from a dict.
* :func:`metric_card`  — render a single card using ``st.metric``.

Import discipline
-----------------
* Only imports ``streamlit`` — no ``ui.*`` imports.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

_PCT_KEYS = ("rate", "pct", "return", "drawdown", "cagr")
_RATIO_KEYS = ("sharpe", "sortino", "calmar")
_INT_KEYS = ("num", "count", "trades")


def _detect_format(key: str) -> str:
    """
    Infer a display format from a metric key name.

    Rules (checked in order)
    ------------------------
    * Key contains ``"rate"``, ``"pct"``, ``"return"``, ``"drawdown"``,
      or ``"cagr"`` → ``"percent"``
    * Key contains ``"sharpe"``, ``"sortino"``, or ``"calmar"`` → ``"ratio"``
    * Key contains ``"num"``, ``"count"``, or ``"trades"`` → ``"integer"``
    * Otherwise → ``"ratio"``

    Parameters
    ----------
    key:
        Lower-cased metric key string.

    Returns
    -------
    str
        One of ``"percent"``, ``"ratio"``, or ``"integer"``.
    """
    lower = key.lower()
    for fragment in _PCT_KEYS:
        if fragment in lower:
            return "percent"
    for fragment in _RATIO_KEYS:
        if fragment in lower:
            return "ratio"
    for fragment in _INT_KEYS:
        if fragment in lower:
            return "integer"
    return "ratio"


def _fmt_value(value: Any, fmt: str) -> str:
    """
    Format *value* according to *fmt*.

    Parameters
    ----------
    value:
        Numeric value to format.  Returns ``"—"`` for ``None``.
    fmt:
        One of ``"percent"``, ``"ratio"``, ``"integer"``, or ``"currency"``.

    Returns
    -------
    str
        Formatted string representation.
    """
    if value is None:
        return "—"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)

    if fmt == "percent":
        return f"{f * 100:.1f}%"
    if fmt == "integer":
        return str(int(round(f)))
    if fmt == "currency":
        return f"${f:,.2f}"
    # default: ratio
    return f"{f:.2f}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def metric_card(
    label: str,
    value: Any,
    delta: Any = None,
    format: str = "auto",
) -> None:
    """
    Render a single metric inside a ``st.metric`` call wrapped in a card div.

    The card inherits the ``.metric-card`` CSS class injected by
    ``ui/app.py``'s global stylesheet.

    Parameters
    ----------
    label:
        Human-readable metric name.  Underscores are replaced with spaces and
        the string is title-cased for display.
    value:
        Numeric (or ``None``) metric value.
    delta:
        Optional delta value displayed below the main value.  Formatted with
        the same *format* rule as *value*.
    format:
        Display format: ``"auto"`` (detect from *label*), ``"percent"``,
        ``"ratio"``, ``"integer"``, or ``"currency"``.
    """
    fmt = _detect_format(label) if format == "auto" else format
    display_label = label.replace("_", " ").title()
    display_value = _fmt_value(value, fmt)
    display_delta = _fmt_value(delta, fmt) if delta is not None else None

    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric(
        label=display_label,
        value=display_value,
        delta=display_delta,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def metrics_grid(
    metrics: dict[str, Any],
    columns: int = 4,
) -> None:
    """
    Render a responsive grid of metric cards from a flat metrics dict.

    Each key becomes a card label; values are auto-formatted by
    :func:`_detect_format`.  Cards are laid out in rows of *columns* width
    using ``st.columns``.

    Parameters
    ----------
    metrics:
        Flat dict mapping metric key strings to numeric values.
    columns:
        Number of cards per row (default 4).

    Example
    -------
    ::

        metrics_grid({"sharpe_ratio": 1.42, "cagr": 0.18, "max_drawdown": -0.11})
    """
    if not metrics:
        st.info("No metrics available.")
        return

    keys = list(metrics.keys())
    cols = st.columns(columns)
    for i, key in enumerate(keys):
        with cols[i % columns]:
            metric_card(label=key, value=metrics[key], format="auto")
