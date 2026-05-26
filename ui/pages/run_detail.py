"""
Run Detail page — full tearsheet for a single backtest run.

Features
--------
* Header: strategy name, status badge, visibility badge, owner, back button.
* Metrics strip: Sharpe, CAGR, Max DD, Win Rate.
* Equity curve chart (when ``run.result["equity_curve"]`` is present).
* Parameters table (``run.params`` as a two-column DataFrame).
* Visibility control — owner or admin can change visibility when the run
  is not ``"RUNNING"``.
* Comment thread with add-comment form (admin / analyst only).

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
from ui.db import (
    create_comment,
    get_db,
    get_run,
    get_user_by_id,
    list_comments_for_run,
    list_users,
    update_run_visibility,
)

# ---------------------------------------------------------------------------
# Private badge helpers (local — avoids importing ui.components.run_card internals)
# ---------------------------------------------------------------------------

_STATUS_COLORS: dict[str, tuple[str, str]] = {
    "RUNNING": ("#7A4F00", "#FFF3CD"),
    "DONE":    ("#27500A", "#EAF3DE"),
    "FAILED":  ("#791F1F", "#FCEBEB"),
}

_VIS_COLORS: dict[str, tuple[str, str]] = {
    "PRIVATE":  ("#6b6b6b", "#f0f0f0"),
    "TEAM":     ("#0C447C", "#E6F1FB"),
    "FEATURED": ("#7A4F00", "#FFF3CD"),
}


def _badge(text: str, color: str, bg: str) -> str:
    return (
        f'<span style="color:{color}; background:{bg}; padding:2px 8px; '
        f'border-radius:4px; font-size:0.78rem; font-weight:500;">{text}</span>'
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_detail_page() -> None:
    """
    Render the run detail / tearsheet page for the run stored in
    ``st.session_state["selected_run_id"]``.

    Role enforcement
    ----------------
    Accessible to ``"admin"``, ``"analyst"``, and ``"viewer"``.

    Guards
    ------
    * No run ID in session → warning + "Go to Run History" button.
    * Run not found in DB → error message.
    * Non-admin trying to view a private run they don't own → error.

    Sections
    --------
    1. **Header** — strategy name title, status/visibility/owner badges,
       "← Back" button.
    2. **Metrics strip** — 4 ``st.metric`` cards: Sharpe, CAGR, Max DD,
       Win Rate (``"—"`` when no result).
    3. **Equity curve** — :func:`~ui.components.equity_chart.equity_chart`
       rendered when ``run.result["equity_curve"]`` is present.
    4. **Parameters** — ``run.params`` rendered as a two-column DataFrame;
       caption if empty.
    5. **Visibility control** — selectbox + "Update" button; shown only to
       owner or admin when the run is not ``"RUNNING"``.  Admin gets the
       ``"FEATURED"`` option; others see ``"PRIVATE"`` and ``"TEAM"`` only.
    6. **Comment thread** — all existing comments in chronological order.
    7. **Add comment form** — ``st.form`` with ``st.text_area`` and a submit
       button; shown only to ``"admin"`` and ``"analyst"``.
    """
    require_role(["admin", "analyst", "viewer"])
    user = get_current_user()
    assert user is not None

    # ── Guard: run must be selected ────────────────────────────────────────
    run_id: str | None = st.session_state.get("selected_run_id")
    if not run_id:
        st.warning("No run selected. Navigate here from **Run History**.")
        if st.button("← Go to Run History", key="rd_no_id_back"):
            st.session_state["page"] = "run_history"
            st.rerun()
        return

    # ── Fetch data (single session) ────────────────────────────────────────
    with get_db() as db:
        run = get_run(db, run_id)
        if run is None:
            st.error(f"Run not found: {run_id!r}")
            return

        # Role-based access check — admin bypasses
        if user["role"] != "admin":
            if (
                run.owner_id != user["user_id"]
                and run.visibility not in ("TEAM", "FEATURED")
            ):
                st.error("You do not have permission to view this run.")
                return

        owner = get_user_by_id(db, run.owner_id)
        comments = list_comments_for_run(db, run_id)
        all_users = list_users(db)

    owner_name: str = owner.name if owner else f"User {run.owner_id}"
    user_name_by_id: dict[int, str] = {u.id: u.name for u in all_users}

    # ── Header ─────────────────────────────────────────────────────────────
    st.title(f"📋 {run.strategy_name}")
    hdr_info, hdr_btn = st.columns([5, 1])

    sc, sb = _STATUS_COLORS.get(run.status, ("#6b6b6b", "#f0f0f0"))
    vc, vb = _VIS_COLORS.get(run.visibility, ("#6b6b6b", "#f0f0f0"))

    with hdr_info:
        st.markdown(
            f"{_badge(run.status, sc, sb)} &nbsp; "
            f"{_badge(run.visibility, vc, vb)} &nbsp; "
            f'<span style="color:#888; font-size:0.9rem;">by {owner_name}</span>',
            unsafe_allow_html=True,
        )

    with hdr_btn:
        if st.button("← Back", key="rd_back"):
            st.session_state["page"] = "run_history"
            st.rerun()
            return  # stop further rendering after rerun

    st.markdown("---")

    # ── Metrics strip ──────────────────────────────────────────────────────
    st.subheader("📈 Metrics")
    metrics: dict = {}
    if run.result and isinstance(run.result, dict):
        metrics = run.result.get("metrics", {})

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Sharpe", fmt_sharpe(metrics.get("sharpe_ratio")))
    with m2:
        st.metric("CAGR", fmt_pct(metrics.get("cagr")))
    with m3:
        st.metric("Max DD", fmt_pct(metrics.get("max_drawdown")))
    with m4:
        st.metric("Win Rate", fmt_pct(metrics.get("win_rate")))

    # ── Equity curve ───────────────────────────────────────────────────────
    if run.result and isinstance(run.result, dict):
        equity_data = run.result.get("equity_curve")
        if equity_data:
            st.subheader("📉 Equity Curve")
            equity_chart(
                equity_data,
                title=f"{run.strategy_name} — Equity Curve",
            )

    # ── Parameters ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("⚙️ Parameters")
    if run.params:
        params_df = pd.DataFrame(
            [{"Parameter": k, "Value": str(v)} for k, v in run.params.items()]
        )
        st.dataframe(params_df, use_container_width=True, hide_index=True)
    else:
        st.caption("No parameters recorded.")

    # ── Visibility control (owner or admin, not while RUNNING) ─────────────
    can_change: bool = (
        user["role"] == "admin" or run.owner_id == user["user_id"]
    ) and run.status != "RUNNING"

    if can_change:
        st.markdown("---")
        st.subheader("🔒 Visibility")
        vis_options = (
            ["PRIVATE", "TEAM", "FEATURED"]
            if user["role"] == "admin"
            else ["PRIVATE", "TEAM"]
        )
        current_idx = (
            vis_options.index(run.visibility)
            if run.visibility in vis_options
            else 0
        )
        new_vis = st.selectbox(
            "Set visibility",
            options=vis_options,
            index=current_idx,
            key="rd_visibility",
        )
        if new_vis != run.visibility:
            if st.button("Update", key="rd_update_vis"):
                with get_db() as db:
                    update_run_visibility(db, run_id, new_vis)
                st.success("Visibility updated.")
                st.rerun()

    # ── Comment thread ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("💬 Comments")

    if not comments:
        st.caption("No comments yet.")
    else:
        for comment in comments:
            author_name = user_name_by_id.get(
                comment.author_id, f"User {comment.author_id}"
            )
            st.markdown(
                f"**{author_name}** · "
                f'<span style="color:#888; font-size:0.85rem;">'
                f"{comment.created_at.strftime('%Y-%m-%d %H:%M')}</span>",
                unsafe_allow_html=True,
            )
            st.markdown(comment.body)

    # ── Add-comment form (admin + analyst only) ────────────────────────────
    if user["role"] in ("admin", "analyst"):
        with st.form(key="rd_comment_form"):
            body = st.text_area("Add a comment", key="rd_comment_body")
            submitted = st.form_submit_button("Post comment")

        if submitted and body.strip():
            with get_db() as db:
                create_comment(
                    db,
                    run_id=run_id,
                    author_id=user["user_id"],
                    body=body.strip(),
                    mentions=[],
                )
            st.rerun()
