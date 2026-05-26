"""
New Run page — strategy selection, parameter configuration, and run submission.

Features
--------
* Two-column layout: strategy picker (left) with description caption,
  parameter form (right) with capital, date range, and symbols.
* Visibility selector: ``"PRIVATE"``, ``"TEAM"``, or ``"FEATURED"``
  (FEATURED restricted to admin).
* Tags input (comma-separated).
* Submit → inserts a :class:`~ui.db.Run` row with ``status="RUNNING"``
  and navigates to the run-detail page.

Import discipline
-----------------
* Imports from ``ui.auth``, ``ui.db``, and ``ui.components`` only.
* No backtester engine imports.
* No cross-page imports.
"""

from __future__ import annotations

from datetime import date

import streamlit as st

from ui.auth import get_current_user, require_role
from ui.db import create_run, get_db, list_strategies

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def new_run_page() -> None:
    """
    Render the new backtest run submission page.

    Role enforcement
    ----------------
    Accessible to ``"admin"`` and ``"analyst"`` only — viewers are blocked.

    Layout
    ------
    * **Left column** — strategy selectbox + description caption.
    * **Right column** — parameter fields: initial capital ($), start date,
      end date (nested 2-column row), and symbols.
    * **Bottom row** — visibility selector, tags input, launch button.

    Validation (on submit)
    ----------------------
    * At least one symbol must be entered.
    * Start date must be strictly before end date.
    * Non-admin users cannot set visibility to ``"FEATURED"`` — a warning
      is displayed immediately and visibility reverts to ``"TEAM"``.

    Submission
    ----------
    On valid submit:

    1. Calls :func:`~ui.db.create_run` with ``status="RUNNING"``.
    2. Stores the new run's id in ``st.session_state["selected_run_id"]``.
    3. Sets ``st.session_state["page"] = "run_detail"``.
    4. Calls ``st.rerun()``.
    """
    require_role(["admin", "analyst"])
    user = get_current_user()
    assert user is not None

    st.title("🚀 New Backtest Run")

    # ── Fetch approved strategies ──────────────────────────────────────────
    with get_db() as db:
        strategies = list_strategies(db, status="APPROVED")

    if not strategies:
        st.warning(
            "No approved strategies available. "
            "Submit one via **Strategy Library** or ask an admin to approve "
            "an existing strategy."
        )
        return

    # ── Two-column layout: strategy picker | parameter form ────────────────
    left, right = st.columns([1, 2])

    with left:
        st.subheader("Strategy")
        strategy_names = [s.name for s in strategies]
        selected_name = st.selectbox(
            "Select strategy",
            options=strategy_names,
            key="nr_strategy",
        )
        selected_strategy = next(
            (s for s in strategies if s.name == selected_name),
            strategies[0],
        )
        st.caption(selected_strategy.description)

    with right:
        st.subheader("Parameters")

        initial_capital = st.number_input(
            "Initial Capital ($)",
            min_value=1_000,
            value=100_000,
            step=1_000,
            key="nr_capital",
        )

        dc1, dc2 = st.columns(2)
        with dc1:
            start_date = st.date_input(
                "Start Date",
                value=date(2020, 1, 1),
                key="nr_start",
            )
        with dc2:
            end_date = st.date_input(
                "End Date",
                value=date(2023, 12, 31),
                key="nr_end",
            )

        symbols_raw = st.text_input(
            "Symbols (comma-separated)",
            value="AAPL,MSFT",
            key="nr_symbols",
        )

    # ── Visibility, tags, submit ───────────────────────────────────────────
    st.markdown("---")
    vis_col, tag_col, btn_col = st.columns([2, 3, 1])

    with vis_col:
        visibility = st.selectbox(
            "Visibility",
            options=["PRIVATE", "TEAM", "FEATURED"],
            key="nr_visibility",
        )
        if user["role"] != "admin" and visibility == "FEATURED":
            st.warning(
                "Only admins can mark a run as FEATURED. Reverting to TEAM."
            )
            visibility = "TEAM"

    with tag_col:
        tags_raw = st.text_input(
            "Tags (comma-separated)",
            value="",
            key="nr_tags",
        )

    with btn_col:
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.button("🚀 Launch", key="nr_submit")

    # ── Submission handler ─────────────────────────────────────────────────
    if not submitted:
        return

    symbols = [s.strip().upper() for s in symbols_raw.split(",") if s.strip()]
    tags = [t.strip() for t in tags_raw.split(",") if t.strip()]

    if not symbols:
        st.error("Please enter at least one symbol.")
        return

    if start_date >= end_date:
        st.error("Start date must be strictly before end date.")
        return

    params: dict = {
        "initial_capital": float(initial_capital),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "symbols": symbols,
    }

    with get_db() as db:
        run = create_run(
            db,
            owner_id=user["user_id"],
            strategy_name=selected_name,
            params=params,
            visibility=visibility,
            tags=tags,
        )
        run_id = run.id

    st.session_state["selected_run_id"] = run_id
    st.session_state["page"] = "run_live"
    st.rerun()
