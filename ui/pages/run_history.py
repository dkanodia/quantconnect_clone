"""
Run History page — filterable, sortable table of backtest runs.

Features
--------
* Filter bar: owner (admin/analyst only), strategy name, status, date range,
  tags — all persisted in ``st.session_state["run_history_filters"]``.
* Results rendered as a ``st.dataframe`` with metric columns extracted from
  ``run.result["metrics"]``.
* Row-click navigation via a selectbox + button below the table.

Import discipline
-----------------
* Imports from ``ui.auth``, ``ui.db``, and ``ui.components`` only.
* No backtester engine imports.
* No cross-page imports.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd
import streamlit as st

from ui.auth import get_current_user, require_role
from ui.db import Run, get_db, list_runs_for_user, list_users

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _get_filters() -> dict:
    """Return the current filter dict from session state."""
    return st.session_state.get("run_history_filters", {})


def _save_filters(filters: dict) -> None:
    """Write *filters* back to session state."""
    st.session_state["run_history_filters"] = filters


def _apply_filters(
    runs: list[Run],
    filters: dict,
    user_id_by_name: dict[str, int],
) -> list[Run]:
    """
    Apply all active filters to *runs* and return the matching subset.

    Filtering is done client-side in Python after a single DB fetch.

    Parameters
    ----------
    runs:
        Full list of runs visible to the current user.
    filters:
        Dict of active filter values from session state.
    user_id_by_name:
        Mapping from user display name to user_id (for owner filter).

    Returns
    -------
    list[Run]
        Filtered list in the same order as the input.
    """
    result = runs

    # Strategy name — case-insensitive substring
    strategy_q = filters.get("strategy", "").strip().lower()
    if strategy_q:
        result = [r for r in result if strategy_q in r.strategy_name.lower()]

    # Status
    status_q = filters.get("status", "All")
    if status_q and status_q != "All":
        result = [r for r in result if r.status == status_q]

    # Owner (by name → user_id)
    owner_q = filters.get("owner", "All")
    if owner_q and owner_q != "All":
        target_id = user_id_by_name.get(owner_q)
        if target_id is None:
            return []  # unknown name → no matches
        result = [r for r in result if r.owner_id == target_id]

    # Date range — compare against run.created_at.date()
    from_date: date | None = filters.get("from_date")
    to_date: date | None = filters.get("to_date")
    if from_date is not None:
        result = [
            r for r in result
            if _run_date(r) >= from_date
        ]
    if to_date is not None:
        result = [
            r for r in result
            if _run_date(r) <= to_date
        ]

    # Tag search — substring match against any tag string
    tag_q = filters.get("tag", "").strip().lower()
    if tag_q:
        result = [
            r for r in result
            if any(tag_q in str(t).lower() for t in (r.tags or []))
        ]

    return result


def _run_date(run: Run) -> date:
    """Return the UTC date of *run.created_at* as a ``datetime.date``."""
    dt = run.created_at
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.date()


def _extract_metric(run: Run, key: str) -> float | None:
    """Safely pull a metric value from run.result, returning None if absent."""
    if not run.result or not isinstance(run.result, dict):
        return None
    return run.result.get("metrics", {}).get(key)


def _fmt_sharpe(v: float | None) -> str | None:
    return f"{v:.2f}" if v is not None else None


def _fmt_pct(v: float | None) -> str | None:
    return f"{v * 100:.1f}%" if v is not None else None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_history_page() -> None:
    """
    Render the run history page with filter bar and results table.

    Role enforcement
    ----------------
    Accessible to ``"admin"``, ``"analyst"``, and ``"viewer"``.

    Filter bar
    ----------
    * **Owner** (admin/analyst only) — select a specific user's runs.
    * **Strategy** — case-insensitive substring match.
    * **Status** — ``"All"``, ``"DONE"``, ``"RUNNING"``, or ``"FAILED"``.
    * **From / To** — date-range restriction on ``run.created_at``.
    * **Tag** — substring match against any tag in ``run.tags``.
    * **Clear filters** — resets all filters and reruns.

    All filter values are persisted in
    ``st.session_state["run_history_filters"]`` across reruns.

    Results table
    -------------
    Columns: Strategy, Owner, Sharpe, CAGR, Max DD, Win Rate, Visibility,
    Status, Created.  Numeric columns are formatted; ``None`` cells render
    as empty.

    Navigation
    ----------
    A selectbox + "Open →" button below the table navigates to
    ``run_detail`` for the selected run.
    """
    require_role(["admin", "analyst", "viewer"])
    user = get_current_user()
    assert user is not None

    st.title("🕐 Run History")

    # ── Fetch data ─────────────────────────────────────────────────────────
    with get_db() as db:
        all_runs = list_runs_for_user(db, user["user_id"], user["role"])
        all_users = list_users(db)

    user_name_by_id: dict[int, str] = {u.id: u.name for u in all_users}
    user_id_by_name: dict[str, int] = {u.name: u.id for u in all_users}
    all_user_names: list[str] = [u.name for u in all_users]

    # ── Filter bar ─────────────────────────────────────────────────────────
    filters = _get_filters()

    with st.expander("🔍 Filters", expanded=True):
        fc1, fc2, fc3 = st.columns(3)
        fd1, fd2, fc4 = st.columns(3)

        with fc1:
            strategy_val = st.text_input(
                "Strategy name contains",
                value=filters.get("strategy", ""),
                key="rh_strategy",
            )

        with fc2:
            status_options = ["All", "DONE", "RUNNING", "FAILED"]
            current_status = filters.get("status", "All")
            status_idx = (
                status_options.index(current_status)
                if current_status in status_options
                else 0
            )
            status_val = st.selectbox(
                "Status",
                options=status_options,
                index=status_idx,
                key="rh_status",
            )

        with fc3:
            # Owner filter only meaningful for admin/analyst
            owner_options = ["All"] + all_user_names
            current_owner = filters.get("owner", "All")
            owner_idx = (
                owner_options.index(current_owner)
                if current_owner in owner_options
                else 0
            )
            if user["role"] in ("admin", "analyst"):
                owner_val = st.selectbox(
                    "Owner",
                    options=owner_options,
                    index=owner_idx,
                    key="rh_owner",
                )
            else:
                owner_val = "All"
                st.empty()  # preserve column layout

        with fd1:
            raw_from = filters.get("from_date")
            from_val = st.date_input(
                "From",
                value=raw_from if isinstance(raw_from, date) else None,
                key="rh_from",
            )

        with fd2:
            raw_to = filters.get("to_date")
            to_val = st.date_input(
                "To",
                value=raw_to if isinstance(raw_to, date) else None,
                key="rh_to",
            )

        with fc4:
            tag_val = st.text_input(
                "Search tags",
                value=filters.get("tag", ""),
                key="rh_tag",
            )

        if st.button("Clear filters", key="rh_clear"):
            _save_filters({})
            st.rerun()
            return  # st.rerun() raises in real runtime; return keeps tests clean

    # Persist updated filters
    _save_filters({
        "strategy": strategy_val,
        "status": status_val,
        "owner": owner_val,
        "from_date": from_val if isinstance(from_val, date) else None,
        "to_date": to_val if isinstance(to_val, date) else None,
        "tag": tag_val,
    })

    # ── Apply filters ──────────────────────────────────────────────────────
    filtered_runs = _apply_filters(all_runs, _get_filters(), user_id_by_name)

    st.caption(f"Showing {len(filtered_runs)} run{'s' if len(filtered_runs) != 1 else ''}")

    # ── Build display DataFrame ────────────────────────────────────────────
    rows = []
    for run in filtered_runs:
        rows.append({
            "Strategy":   run.strategy_name,
            "Owner":      user_name_by_id.get(run.owner_id, f"User {run.owner_id}"),
            "Sharpe":     _fmt_sharpe(_extract_metric(run, "sharpe_ratio")),
            "CAGR":       _fmt_pct(_extract_metric(run, "cagr")),
            "Max DD":     _fmt_pct(_extract_metric(run, "max_drawdown")),
            "Win Rate":   _fmt_pct(_extract_metric(run, "win_rate")),
            "Visibility": run.visibility,
            "Status":     run.status,
            "Created":    run.created_at.strftime("%Y-%m-%d %H:%M"),
        })

    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["Strategy", "Owner", "Sharpe", "CAGR", "Max DD",
                 "Win Rate", "Visibility", "Status", "Created"]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Detail navigation ──────────────────────────────────────────────────
    if filtered_runs:
        st.markdown("---")
        nav_col, btn_col = st.columns([5, 1])
        with nav_col:
            selected_id = st.selectbox(
                "View run detail",
                options=[r.id for r in filtered_runs],
                format_func=lambda rid: next(
                    (r.strategy_name for r in filtered_runs if r.id == rid),
                    rid,
                ),
                key="rh_select_run",
            )
        with btn_col:
            st.markdown("<br>", unsafe_allow_html=True)  # vertical align
            if st.button("Open →", key="rh_open_run"):
                st.session_state["selected_run_id"] = selected_id
                st.session_state["page"] = "run_detail"
                st.rerun()
