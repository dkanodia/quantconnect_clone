"""
Reusable run summary card component.

Exposes a single public function :func:`run_card` that renders one
:class:`~ui.db.Run` as a bordered card.  Clicking the strategy-name
button calls the caller-supplied ``on_click`` callback with the run UUID.

Import discipline
-----------------
* Imports from ``ui.db`` only — never from pages or other components.
* Opens its own ``get_db()`` session to resolve the owner name when the
  owner is not the current user.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

import streamlit as st

from ui.db import Run, get_db, get_user_by_id

# ---------------------------------------------------------------------------
# Badge colour palettes
# ---------------------------------------------------------------------------

_VISIBILITY_COLORS: dict[str, tuple[str, str]] = {
    "PRIVATE":  ("#6b6b6b", "#f0f0f0"),
    "TEAM":     ("#0C447C", "#E6F1FB"),
    "FEATURED": ("#7A4F00", "#FFF3CD"),
}

_STATUS_COLORS: dict[str, tuple[str, str]] = {
    "RUNNING": ("#7A4F00", "#FFF3CD"),
    "DONE":    ("#27500A", "#EAF3DE"),
    "FAILED":  ("#791F1F", "#FCEBEB"),
}

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _badge(text: str, color: str, bg: str) -> str:
    """Return an inline-HTML badge ``<span>``."""
    return (
        f'<span style="color:{color}; background:{bg}; padding:2px 8px; '
        f'border-radius:4px; font-size:0.78rem; font-weight:500;">{text}</span>'
    )


def _derive_initials(name: str) -> str:
    """Return up to 4 uppercase initials from a display name."""
    words = name.strip().split()
    return "".join(w[0].upper() for w in words if w)[:4] or "?"


def fmt_sharpe(v: float | None) -> str:
    """Format a Sharpe ratio to 2 decimal places, or ``"—"`` if absent."""
    return f"{v:.2f}" if v is not None else "—"


def fmt_pct(v: float | None) -> str:
    """Format a 0–1 fraction as a percentage string, or ``"—"`` if absent."""
    return f"{v * 100:.1f}%" if v is not None else "—"


def time_ago(dt: datetime) -> str:
    """
    Return a human-readable relative time string.

    Examples: ``"just now"``, ``"4m ago"``, ``"2h ago"``, ``"3d ago"``.
    Naive datetimes are assumed to be UTC.
    """
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = now - dt
    total_secs = int(delta.total_seconds())
    if total_secs < 60:
        return "just now"
    if total_secs < 3_600:
        return f"{total_secs // 60}m ago"
    if total_secs < 86_400:
        return f"{total_secs // 3_600}h ago"
    return f"{delta.days}d ago"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_card(
    run: Run,
    current_user: dict,
    on_click: Callable[[str], None],
) -> None:
    """
    Render a single backtest run as a bordered summary card.

    Layout (top to bottom)
    ----------------------
    1. **Top row** — strategy name button, visibility badge, status badge.
    2. **Metric pills** — Sharpe, CAGR, Max DD, Win Rate (``"—"`` when no
       result yet).
    3. **Bottom row** — owner avatar circle + name, relative timestamp, tag
       pills.

    Clicking the strategy-name button calls ``on_click(run.id)`` so the
    caller can navigate to the run-detail page.

    Parameters
    ----------
    run:
        The :class:`~ui.db.Run` ORM instance to display.  Column-mapped
        attributes are accessible after the originating session closes.
    current_user:
        Auth dict returned by :func:`~ui.auth.get_current_user`.
    on_click:
        Zero-or-one-argument callable invoked with the run UUID string
        when the user clicks the card's strategy-name button.
    """
    # ── Resolve owner display info ─────────────────────────────────────────
    if run.owner_id == current_user["user_id"]:
        owner_name: str = current_user["name"]
        owner_initials: str = current_user["avatar_initials"]
    else:
        with get_db() as db:
            owner = get_user_by_id(db, run.owner_id)
        if owner is not None:
            owner_name = owner.name
            owner_initials = _derive_initials(owner.name)
        else:
            owner_name = f"User {run.owner_id}"
            owner_initials = "?"

    # ── Extract metrics ────────────────────────────────────────────────────
    metrics_data: dict = {}
    if run.result and isinstance(run.result, dict):
        metrics_data = run.result.get("metrics", {})

    sharpe_str = fmt_sharpe(metrics_data.get("sharpe_ratio"))
    cagr_str = fmt_pct(metrics_data.get("cagr"))
    max_dd_str = fmt_pct(metrics_data.get("max_drawdown"))
    win_rate_str = fmt_pct(metrics_data.get("win_rate"))

    # ── Build badge HTML ───────────────────────────────────────────────────
    vis_color, vis_bg = _VISIBILITY_COLORS.get(
        run.visibility, ("#6b6b6b", "#f0f0f0")
    )
    st_color, st_bg = _STATUS_COLORS.get(run.status, ("#6b6b6b", "#f0f0f0"))
    vis_badge_html = _badge(run.visibility, vis_color, vis_bg)
    status_badge_html = _badge(run.status, st_color, st_bg)

    # ── Tag pills HTML ─────────────────────────────────────────────────────
    tag_html = " ".join(
        f'<span style="background:#f0f0f0; color:#555; padding:1px 6px; '
        f'border-radius:4px; font-size:0.75rem;">{t}</span>'
        for t in (run.tags or [])
    )

    # ── Owner avatar HTML ──────────────────────────────────────────────────
    avatar_html = (
        f'<span style="display:inline-flex; align-items:center; '
        f'justify-content:center; width:22px; height:22px; '
        f'border-radius:50%; background:#1a1a1a; color:white; '
        f'font-size:0.6rem; font-weight:600; margin-right:5px; '
        f'vertical-align:middle;">{owner_initials}</span>'
        f'<span style="font-size:0.85rem; color:#444; '
        f'vertical-align:middle;">{owner_name}</span>'
    )

    # ── Render card ────────────────────────────────────────────────────────
    with st.container(border=True):
        # Top row: name button + badges
        top_name, top_badges = st.columns([3, 2])
        with top_name:
            if st.button(
                run.strategy_name,
                key=f"run_card_{run.id}",
                use_container_width=True,
            ):
                on_click(run.id)
        with top_badges:
            st.markdown(
                f"{vis_badge_html} &nbsp; {status_badge_html}",
                unsafe_allow_html=True,
            )

        # Metric pills
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Sharpe", sharpe_str)
        m2.metric("CAGR", cagr_str)
        m3.metric("Max DD", max_dd_str)
        m4.metric("Win Rate", win_rate_str)

        # Bottom row: owner + time + tags
        bot_left, bot_right = st.columns([4, 1])
        with bot_left:
            st.markdown(
                f"{avatar_html}"
                + (f" &nbsp; {tag_html}" if tag_html else ""),
                unsafe_allow_html=True,
            )
        with bot_right:
            st.markdown(
                f'<span style="color:#888; font-size:0.8rem; '
                f'float:right;">{time_ago(run.created_at)}</span>',
                unsafe_allow_html=True,
            )
