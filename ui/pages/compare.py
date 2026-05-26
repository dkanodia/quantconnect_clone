"""
Compare Runs page — side-by-side metric and equity-curve comparison of 2–4 runs.

Features
--------
* Multiselect picker — uses every run visible to the current user.
  Pre-populates from ``st.session_state["compare_run_ids"]`` if set.
* Validates 2–4 selections; warns and truncates if > 4 chosen.
* Metric comparison table — Strategy, Status, Sharpe, CAGR, Max DD, Win Rate.
* Equity curve overlay — primary curve + optional compare curves rendered via
  :func:`~ui.components.equity_chart.equity_chart`; shown only when at least
  one selected run has ``result["equity_curve"]`` data.

Import discipline
-----------------
* Imports from ``ui.auth``, ``ui.db``, and ``ui.components`` only.
* No backtester engine imports.
* No cross-page imports.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ui.auth import get_current_user, require_role
from ui.components.equity_chart import equity_chart
from ui.components.run_card import fmt_pct, fmt_sharpe
from ui.db import get_db, list_runs_for_user

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def compare_page() -> None:
    """
    Render the side-by-side run comparison page.

    Role enforcement
    ----------------
    Accessible to ``"admin"``, ``"analyst"``, and ``"viewer"``.

    Selection
    ---------
    A ``st.multiselect`` lists every run visible to the current user (by id,
    displayed as strategy name).  Pre-selected ids are taken from
    ``st.session_state["compare_run_ids"]`` (filtered to ids that are still
    visible to this user).

    Validation
    ----------
    * Fewer than 2 runs selected → ``st.info`` and early return.
    * More than 4 runs selected → ``st.warning`` and truncation to first 4.

    Sections (after valid selection)
    ---------------------------------
    1. **Metric comparison table** — one row per run, columns: Strategy,
       Status, Sharpe, CAGR, Max DD, Win Rate.
    2. **Equity curve overlay** — :func:`~ui.components.equity_chart.equity_chart`
       with the first run as the primary curve and subsequent runs as
       ``compare_curves``; shown only when at least one run has
       ``result["equity_curve"]``.  Caption shown when no equity data.
    """
    require_role(["admin", "analyst", "viewer"])
    user = get_current_user()
    assert user is not None

    st.title("⚖️ Compare Runs")

    # ── Fetch all runs visible to current user ─────────────────────────────
    with get_db() as db:
        all_visible = list_runs_for_user(db, user["user_id"], user["role"])

    run_by_id: dict[str, object] = {r.id: r for r in all_visible}

    preselected = [
        rid for rid in st.session_state.get("compare_run_ids", [])
        if rid in run_by_id
    ]

    # ── Run selector ───────────────────────────────────────────────────────
    selected_ids: list[str] = st.multiselect(
        "Select runs to compare (2–4)",
        options=[r.id for r in all_visible],
        default=preselected,
        format_func=lambda rid: (
            run_by_id[rid].strategy_name if rid in run_by_id else rid  # type: ignore[union-attr]
        ),
        key="cp_run_ids",
    )

    if len(selected_ids) < 2:
        st.info("Select at least 2 runs to compare.")
        return

    if len(selected_ids) > 4:
        st.warning(
            "Compare supports up to 4 runs. Only the first 4 will be shown."
        )
        selected_ids = selected_ids[:4]

    selected_runs = [run_by_id[rid] for rid in selected_ids if rid in run_by_id]  # type: ignore[assignment]

    # ── Metric comparison table ────────────────────────────────────────────
    st.subheader("📊 Metric Comparison")
    rows = []
    for run in selected_runs:
        metrics: dict = {}
        if run.result and isinstance(run.result, dict):  # type: ignore[union-attr]
            metrics = run.result.get("metrics", {})  # type: ignore[union-attr]
        rows.append({
            "Strategy": run.strategy_name,  # type: ignore[union-attr]
            "Status":   run.status,  # type: ignore[union-attr]
            "Sharpe":   fmt_sharpe(metrics.get("sharpe_ratio")),
            "CAGR":     fmt_pct(metrics.get("cagr")),
            "Max DD":   fmt_pct(metrics.get("max_drawdown")),
            "Win Rate": fmt_pct(metrics.get("win_rate")),
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Equity curve overlay ───────────────────────────────────────────────
    runs_with_equity = [
        r for r in selected_runs
        if r.result and isinstance(r.result, dict) and r.result.get("equity_curve")  # type: ignore[union-attr]
    ]

    if runs_with_equity:
        st.subheader("📉 Equity Curves")
        primary = runs_with_equity[0]
        compare_curves = [
            (r.strategy_name, r.result["equity_curve"])  # type: ignore[index]
            for r in runs_with_equity[1:]
        ] or None
        equity_chart(
            primary.result["equity_curve"],  # type: ignore[index]
            title="Equity Curve Comparison",
            compare_curves=compare_curves,
        )
    else:
        st.caption("No equity curve data available for the selected runs.")
