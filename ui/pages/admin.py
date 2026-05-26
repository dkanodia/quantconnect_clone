"""
Admin Panel page — user management, strategy review queue, and system stats.

Features
--------
* System stats strip — total users, total runs, active runs, pending strategies.
* User management — all non-self users listed with role-change selectbox and
  "Save" button; calls :func:`~ui.db.update_user_role` on click.
* Strategy review queue — PENDING strategies in expanders with "✅ Approve"
  and "❌ Reject" buttons.

Import discipline
-----------------
* Imports from ``ui.auth``, ``ui.db``, and ``ui.components`` only.
* No backtester engine imports.
* No cross-page imports.
"""

from __future__ import annotations

import streamlit as st

from ui.auth import get_current_user, require_role
from ui.db import (
    get_db,
    list_runs_for_user,
    list_strategies,
    list_users,
    update_strategy_status,
    update_user_role,
)

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def admin_page() -> None:
    """
    Render the admin panel.

    Role enforcement
    ----------------
    Accessible to ``"admin"`` **only**.

    Sections
    --------
    1. **System overview** — 4 ``st.metric`` cards:

       * Total Users
       * Total Runs
       * Active Runs (status ``"RUNNING"``)
       * Pending Strategies

    2. **User management** — one row per non-self user containing the
       user's name + email, a role selectbox (``"viewer"``, ``"analyst"``,
       ``"admin"``), and a "Save" button that appears only when the
       selected role differs from the current role.  Clicking "Save" calls
       :func:`~ui.db.update_user_role` and reruns.

    3. **Strategy review queue** — PENDING strategies in ``st.expander``
       cards.  Each card has "✅ Approve" and "❌ Reject" buttons; Approve
       sets status to ``"APPROVED"``, Reject reverts to ``"DRAFT"``.
    """
    require_role(["admin"])
    user = get_current_user()
    assert user is not None

    st.title("⚙️ Admin Panel")

    # ── Fetch data (single session) ────────────────────────────────────────
    with get_db() as db:
        all_users = list_users(db)
        all_runs = list_runs_for_user(db, user["user_id"], user["role"])
        pending_strategies = list_strategies(db, status="PENDING")

    # ── 1. System stats ────────────────────────────────────────────────────
    st.subheader("📊 System Overview")
    active_runs = sum(1 for r in all_runs if r.status == "RUNNING")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Users", len(all_users))
    with c2:
        st.metric("Total Runs", len(all_runs))
    with c3:
        st.metric("Active Runs", active_runs)
    with c4:
        st.metric("Pending Strategies", len(pending_strategies))

    # ── 2. User management ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("👥 User Management")

    _roles = ["viewer", "analyst", "admin"]

    for user_row in all_users:
        if user_row.id == user["user_id"]:
            continue  # admin cannot change their own role here

        cols = st.columns([3, 2, 1])
        with cols[0]:
            st.markdown(f"**{user_row.name}** — {user_row.email}")
        with cols[1]:
            current_idx = _roles.index(user_row.role) if user_row.role in _roles else 0
            new_role = st.selectbox(
                "Role",
                options=_roles,
                index=current_idx,
                key=f"adm_role_{user_row.id}",
                label_visibility="collapsed",
            )
        with cols[2]:
            if new_role != user_row.role:
                if st.button("Save", key=f"adm_save_{user_row.id}"):
                    with get_db() as db:
                        update_user_role(db, user_row.id, new_role)
                    st.success(f"Role updated for {user_row.name}.")
                    st.rerun()

    # ── 3. Strategy review queue ───────────────────────────────────────────
    st.markdown("---")
    st.subheader("📋 Strategy Review Queue")

    user_name_by_id: dict[int, str] = {u.id: u.name for u in all_users}

    if not pending_strategies:
        st.info("No strategies pending review.")
    else:
        for strat in pending_strategies:
            author_name = user_name_by_id.get(
                strat.author_id, f"User {strat.author_id}"
            )
            with st.expander(f"**{strat.name}** — by {author_name}"):
                st.markdown(strat.description)
                st.code(strat.code, language="python")

                approve_col, reject_col = st.columns(2)
                with approve_col:
                    if st.button(
                        "✅ Approve", key=f"adm_approve_{strat.id}"
                    ):
                        with get_db() as db:
                            update_strategy_status(db, strat.id, "APPROVED")
                        st.success(f"'{strat.name}' approved.")
                        st.rerun()
                with reject_col:
                    if st.button(
                        "❌ Reject", key=f"adm_reject_{strat.id}"
                    ):
                        with get_db() as db:
                            update_strategy_status(db, strat.id, "DRAFT")
                        st.warning(f"'{strat.name}' returned to draft.")
                        st.rerun()
