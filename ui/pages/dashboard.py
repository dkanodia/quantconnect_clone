"""
Dashboard page — team overview, featured runs, metrics strip, and activity feed.

Sections (top to bottom)
-------------------------
1. Header with current date.
2. Featured runs — admin-pinned runs as a card grid.
3. Team metrics strip — total runs, best Sharpe, avg CAGR, active runs.
4. Today's activity feed — runs created today (UTC).
5. My recent runs — the current user's 5 most recent runs.

Import discipline
-----------------
* Imports from ``ui.auth``, ``ui.db``, and ``ui.components`` only.
* No backtester engine imports.
* No cross-page imports.
"""

from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st

from ui.auth import get_current_user, require_role
from ui.components.run_card import run_card
from ui.db import get_db, list_featured_runs, list_runs_for_user, get_user_by_id

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _navigate_to_run(run_id: str) -> None:
    """
    Navigate to the run-detail page for *run_id*.

    Sets the two relevant session-state keys and calls ``st.rerun()``.
    """
    st.session_state["selected_run_id"] = run_id
    st.session_state["page"] = "run_detail"
    st.rerun()


def _status_badge_html(status: str) -> str:
    """Return a small coloured HTML badge for a run status string."""
    _colors = {
        "RUNNING": ("#7A4F00", "#FFF3CD"),
        "DONE":    ("#27500A", "#EAF3DE"),
        "FAILED":  ("#791F1F", "#FCEBEB"),
    }
    color, bg = _colors.get(status, ("#6b6b6b", "#f0f0f0"))
    return (
        f'<span style="color:{color}; background:{bg}; padding:1px 7px; '
        f'border-radius:4px; font-size:0.78rem; font-weight:500;">{status}</span>'
    )


def _time_ago_short(dt: datetime) -> str:
    """Return a compact relative-time string for the activity feed."""
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = now - dt
    secs = int(delta.total_seconds())
    if secs < 3_600:
        return f"{secs // 60}m ago"
    if secs < 86_400:
        return f"{secs // 3_600}h ago"
    return f"{delta.days}d ago"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def dashboard_page() -> None:
    """
    Render the team dashboard page.

    Role enforcement
    ----------------
    Accessible to ``"admin"``, ``"analyst"``, and ``"viewer"``.

    Sections
    --------
    1. **Header** — title + today's date.
    2. **Featured runs** — up to 9 cards in a 3-column grid; info message
       when none exist.
    3. **Team metrics strip** — 4 ``st.metric`` cards: total team runs, best
       Sharpe, average CAGR, active (RUNNING) runs.
    4. **Today's activity** — up to 10 runs created today (UTC) visible to
       the current user, rendered as a bulleted list.
    5. **My recent runs** — the current user's 5 most recent runs as
       :func:`~ui.components.run_card.run_card` cards.
    """
    require_role(["admin", "analyst", "viewer"])
    user = get_current_user()
    assert user is not None  # guaranteed by require_role

    # ── 1. Header ──────────────────────────────────────────────────────────
    st.title("📊 Team Dashboard")
    today_str = datetime.now(timezone.utc).strftime("%B %d, %Y")
    st.caption(f"Today is {today_str} (UTC)")

    # ── Fetch data (single session) ────────────────────────────────────────
    with get_db() as db:
        featured_runs = list_featured_runs(db)
        visible_runs = list_runs_for_user(db, user["user_id"], user["role"])
        # Resolve owner names for the activity feed
        owner_names: dict[int, str] = {}
        for run in visible_runs:
            if run.owner_id not in owner_names:
                owner = get_user_by_id(db, run.owner_id)
                owner_names[run.owner_id] = owner.name if owner else f"User {run.owner_id}"

    # ── 2. Featured runs ───────────────────────────────────────────────────
    st.subheader("⭐ Featured Runs")
    if not featured_runs:
        st.info("No featured runs yet.")
    else:
        cols = st.columns(3)
        for i, run in enumerate(featured_runs[:9]):
            with cols[i % 3]:
                run_card(run, user, on_click=_navigate_to_run)

    st.markdown("---")

    # ── 3. Team metrics strip ──────────────────────────────────────────────
    st.subheader("📈 Team Metrics")
    team_runs = [
        r for r in visible_runs if r.visibility in ("TEAM", "FEATURED")
    ]
    runs_with_result = [r for r in team_runs if r.result]
    sharpe_values = [
        r.result["metrics"]["sharpe_ratio"]
        for r in runs_with_result
        if isinstance(r.result, dict)
        and "metrics" in r.result
        and "sharpe_ratio" in r.result["metrics"]
    ]
    cagr_values = [
        r.result["metrics"]["cagr"]
        for r in runs_with_result
        if isinstance(r.result, dict)
        and "metrics" in r.result
        and "cagr" in r.result["metrics"]
    ]
    active_count = sum(1 for r in visible_runs if r.status == "RUNNING")
    best_sharpe = max(sharpe_values) if sharpe_values else None
    avg_cagr = (sum(cagr_values) / len(cagr_values)) if cagr_values else None

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Team Runs", len(team_runs))
    with m2:
        st.metric(
            "Best Sharpe",
            f"{best_sharpe:.2f}" if best_sharpe is not None else "—",
        )
    with m3:
        st.metric(
            "Avg CAGR",
            f"{avg_cagr * 100:.1f}%" if avg_cagr is not None else "—",
        )
    with m4:
        st.metric("Active Runs", active_count)

    st.markdown("---")

    # ── 4. Today's activity feed ───────────────────────────────────────────
    st.subheader("🕐 Today's Activity")
    today_utc = datetime.now(timezone.utc).date()
    today_runs = [
        r for r in visible_runs
        if (
            r.created_at.replace(tzinfo=timezone.utc)
            if r.created_at.tzinfo is None
            else r.created_at
        ).date() == today_utc
    ]
    today_runs_sorted = sorted(
        today_runs, key=lambda r: r.created_at, reverse=True
    )[:10]

    if not today_runs_sorted:
        st.caption("No runs today.")
    else:
        for run in today_runs_sorted:
            owner = owner_names.get(run.owner_id, f"User {run.owner_id}")
            badge_html = _status_badge_html(run.status)
            ago = _time_ago_short(run.created_at)
            st.markdown(
                f"**{owner}** ran **{run.strategy_name}** — "
                f"{badge_html} &nbsp; <span style='color:#888;font-size:0.85rem;'>{ago}</span>",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── 5. My recent runs ──────────────────────────────────────────────────
    st.subheader("🏃 My Recent Runs")
    my_runs = sorted(
        [r for r in visible_runs if r.owner_id == user["user_id"]],
        key=lambda r: r.created_at,
        reverse=True,
    )[:5]

    if not my_runs:
        st.info("No runs yet — start one from **New Run**.")
    else:
        for run in my_runs:
            run_card(run, user, on_click=_navigate_to_run)
