"""
Dashboard page — team overview, featured runs, metrics strip, charts, and
activity feed.

Sections (top to bottom)
-------------------------
1. Header with current date.
2. KPI strip — two rows of four metrics each.
3. Charts row — Risk/Return scatter (Sharpe vs CAGR) and Team Activity bar.
4. Leaderboard — top-10 runs by Sharpe ratio.
5. Featured runs — admin-pinned runs as a card grid.
6. Today's activity feed — runs created today (UTC).
7. My recent runs — the current user's 5 most recent runs.

Import discipline
-----------------
* Imports from ``ui.auth``, ``ui.db``, and ``ui.components`` only.
* No backtester engine imports.
* No cross-page imports.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.graph_objects as go
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
    """Return a small coloured HTML badge for a run status string.

    Uses rgba backgrounds so badges render correctly in both light and dark
    Streamlit themes.
    """
    _styles = {
        "RUNNING": "color:#b07d00; background:rgba(176,125,0,0.14);",
        "DONE":    "color:#3a7d1e; background:rgba(58,125,30,0.14);",
        "FAILED":  "color:#c0392b; background:rgba(192,57,43,0.14);",
    }
    style = _styles.get(status, "opacity:0.6;")
    return (
        f'<span style="{style} padding:1px 7px; '
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


def _scatter_chart(runs_with_result: list, owner_names: dict) -> go.Figure:
    """
    Return a Sharpe vs CAGR scatter figure.

    Marker size scales with win rate; colour encodes CAGR on a
    red→yellow→green scale.  Transparent background for theme compatibility.
    """
    xs, ys, sizes, colors, texts = [], [], [], [], []
    for r in runs_with_result:
        m = r.result.get("metrics", {}) if isinstance(r.result, dict) else {}
        cagr = m.get("cagr")
        sharpe = m.get("sharpe_ratio")
        win_rate = m.get("win_rate") or 0.0
        if cagr is None or sharpe is None:
            continue
        xs.append(cagr)
        ys.append(sharpe)
        sizes.append(max(10, win_rate * 45))
        colors.append(cagr)
        owner = owner_names.get(r.owner_id, f"User {r.owner_id}")
        texts.append(
            f"<b>{r.strategy_name}</b><br>"
            f"Owner: {owner}<br>"
            f"CAGR: {cagr * 100:.1f}%<br>"
            f"Sharpe: {sharpe:.2f}<br>"
            f"Win Rate: {win_rate * 100:.1f}%"
        )

    fig = go.Figure(go.Scatter(
        x=xs,
        y=ys,
        mode="markers",
        marker=dict(
            size=sizes,
            color=colors,
            colorscale="RdYlGn",
            showscale=True,
            colorbar=dict(
                title="CAGR",
                tickformat=".0%",
                thickness=12,
                len=0.8,
            ),
            line=dict(width=0),
            opacity=0.85,
        ),
        text=texts,
        hovertemplate="%{text}<extra></extra>",
    ))
    fig.update_layout(
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=60, t=16, b=0),
        showlegend=False,
        xaxis=dict(
            title="CAGR",
            tickformat=".0%",
            gridcolor="rgba(128,128,128,0.12)",
            zeroline=True,
            zerolinecolor="rgba(128,128,128,0.3)",
            zerolinewidth=1,
        ),
        yaxis=dict(
            title="Sharpe",
            gridcolor="rgba(128,128,128,0.12)",
            zeroline=True,
            zerolinecolor="rgba(128,128,128,0.3)",
            zerolinewidth=1,
        ),
    )
    return fig


def _activity_bar_chart(runs: list) -> go.Figure:
    """
    Return a bar chart counting runs per day for the last 14 days.

    Days with no activity show as zero-height bars to preserve the date axis.
    """
    today = datetime.now(timezone.utc).date()
    dates = [(today - timedelta(days=i)) for i in range(13, -1, -1)]

    day_counts: Counter = Counter(
        (
            r.created_at.replace(tzinfo=timezone.utc)
            if r.created_at.tzinfo is None
            else r.created_at
        ).date()
        for r in runs
    )
    counts = [day_counts.get(d, 0) for d in dates]
    labels = [f"{d.strftime('%b')} {d.day}" for d in dates]

    fig = go.Figure(go.Bar(
        x=labels,
        y=counts,
        marker_color="#4f6ef7",
        marker_opacity=0.8,
        marker_line_width=0,
        hovertemplate="%{x}: %{y} run(s)<extra></extra>",
    ))
    fig.update_layout(
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=16, b=0),
        showlegend=False,
        bargap=0.3,
        xaxis=dict(
            showgrid=False,
            tickangle=-45,
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            gridcolor="rgba(128,128,128,0.12)",
            dtick=1,
            tickfont=dict(size=10),
        ),
    )
    return fig


def _leaderboard_df(runs_with_result: list, owner_names: dict) -> pd.DataFrame:
    """
    Build a leaderboard DataFrame sorted by Sharpe ratio (descending, top 10).

    Columns: Strategy, Owner, Sharpe, CAGR, Max DD, Win Rate.
    """
    rows = []
    for r in runs_with_result:
        m = r.result.get("metrics", {}) if isinstance(r.result, dict) else {}
        sharpe = m.get("sharpe_ratio")
        if sharpe is None:
            continue
        cagr = m.get("cagr")
        max_dd = m.get("max_drawdown")
        win_rate = m.get("win_rate")
        rows.append({
            "Strategy":  r.strategy_name,
            "Owner":     owner_names.get(r.owner_id, f"User {r.owner_id}"),
            "Sharpe":    f"{sharpe:.2f}",
            "CAGR":      f"{cagr * 100:.1f}%" if cagr is not None else "—",
            "Max DD":    f"{max_dd * 100:.1f}%" if max_dd is not None else "—",
            "Win Rate":  f"{win_rate * 100:.1f}%" if win_rate is not None else "—",
            "_sharpe":   sharpe,   # numeric, used for sorting only
        })

    if not rows:
        return pd.DataFrame()

    df = (
        pd.DataFrame(rows)
        .sort_values("_sharpe", ascending=False)
        .head(10)
        .drop("_sharpe", axis=1)
        .reset_index(drop=True)
    )
    return df


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
    2. **KPI strip (row 1)** — Total Team Runs, Best Sharpe, Avg CAGR,
       Active Runs.
    3. **KPI strip (row 2)** — Best CAGR, Avg Win Rate, Avg Max DD,
       Completed Runs.
    4. **Charts** — Risk/Return scatter (Sharpe vs CAGR, bubble = win rate)
       and Team Activity bar chart (runs per day, last 14 days).
    5. **Leaderboard** — top-10 runs by Sharpe ratio as a formatted
       ``st.dataframe``.
    6. **Featured runs** — up to 9 cards in a 3-column grid.
    7. **Today's activity** — up to 10 runs created today (UTC).
    8. **My recent runs** — the current user's 5 most recent runs.
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
        owner_names: dict[int, str] = {}
        for run in visible_runs:
            if run.owner_id not in owner_names:
                owner = get_user_by_id(db, run.owner_id)
                owner_names[run.owner_id] = (
                    owner.name if owner else f"User {run.owner_id}"
                )

    # Pre-compute shared metric groups
    team_runs = [r for r in visible_runs if r.visibility in ("TEAM", "FEATURED")]
    runs_with_result = [
        r for r in team_runs
        if r.result and isinstance(r.result, dict) and r.status == "DONE"
    ]

    sharpe_values = [
        r.result["metrics"]["sharpe_ratio"]
        for r in runs_with_result
        if "metrics" in r.result and "sharpe_ratio" in r.result["metrics"]
    ]
    cagr_values = [
        r.result["metrics"]["cagr"]
        for r in runs_with_result
        if "metrics" in r.result and "cagr" in r.result["metrics"]
    ]
    win_rate_values = [
        r.result["metrics"]["win_rate"]
        for r in runs_with_result
        if "metrics" in r.result and "win_rate" in r.result["metrics"]
    ]
    max_dd_values = [
        r.result["metrics"]["max_drawdown"]
        for r in runs_with_result
        if "metrics" in r.result and "max_drawdown" in r.result["metrics"]
    ]

    active_count  = sum(1 for r in visible_runs if r.status == "RUNNING")
    best_sharpe   = max(sharpe_values)   if sharpe_values   else None
    avg_cagr      = sum(cagr_values) / len(cagr_values) if cagr_values else None
    best_cagr     = max(cagr_values)     if cagr_values     else None
    avg_win_rate  = sum(win_rate_values) / len(win_rate_values) if win_rate_values else None
    avg_max_dd    = sum(max_dd_values)   / len(max_dd_values)   if max_dd_values   else None
    completed     = sum(1 for r in team_runs if r.status == "DONE")

    # ── 2. KPI strip — row 1 ──────────────────────────────────────────────
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

    # ── 3. KPI strip — row 2 ──────────────────────────────────────────────
    e1, e2, e3, e4 = st.columns(4)
    with e1:
        st.metric(
            "Best CAGR",
            f"{best_cagr * 100:.1f}%" if best_cagr is not None else "—",
        )
    with e2:
        st.metric(
            "Avg Win Rate",
            f"{avg_win_rate * 100:.1f}%" if avg_win_rate is not None else "—",
        )
    with e3:
        st.metric(
            "Avg Max DD",
            f"{avg_max_dd * 100:.1f}%" if avg_max_dd is not None else "—",
        )
    with e4:
        st.metric("Completed", completed)

    st.markdown("---")

    # ── 4. Charts row ──────────────────────────────────────────────────────
    chart_l, chart_r = st.columns(2)

    with chart_l:
        st.subheader("📈 Risk / Return")
        if len(runs_with_result) >= 2:
            st.plotly_chart(
                _scatter_chart(runs_with_result, owner_names),
                use_container_width=True,
            )
        else:
            st.caption(
                "Run at least 2 strategies to see the risk/return scatter."
            )

    with chart_r:
        st.subheader("📅 Team Activity")
        st.plotly_chart(
            _activity_bar_chart(visible_runs),
            use_container_width=True,
        )

    st.markdown("---")

    # ── 5. Leaderboard ─────────────────────────────────────────────────────
    st.subheader("🏆 Leaderboard")
    lb_df = _leaderboard_df(runs_with_result, owner_names)
    if lb_df.empty:
        st.info("No completed runs with results yet — launch a strategy to populate the leaderboard.")
    else:
        st.dataframe(lb_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── 6. Featured runs ───────────────────────────────────────────────────
    st.subheader("⭐ Featured Runs")
    if not featured_runs:
        st.info("No featured runs yet. Admins can pin runs from the run detail page.")
    else:
        cols = st.columns(3)
        for i, run in enumerate(featured_runs[:9]):
            with cols[i % 3]:
                run_card(run, user, on_click=_navigate_to_run)

    st.markdown("---")

    # ── 7. Today's activity feed ───────────────────────────────────────────
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
                f"{badge_html} &nbsp;"
                f"<span style='opacity:0.55; font-size:0.85rem;'>{ago}</span>",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── 8. My recent runs ──────────────────────────────────────────────────
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
