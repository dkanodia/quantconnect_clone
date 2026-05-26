"""
Strategy Library page — browse, submit, and approve trading strategies.

Features
--------
* Search bar — case-insensitive substring match on name and description.
* Status filter — admin sees All / APPROVED / PENDING / DRAFT; other roles
  always see only APPROVED strategies.
* Strategy cards (``st.expander``) — name, author, status, description,
  Python code, and a "🚀 Run This" button.
* Admin-only approve button — shown for PENDING strategies; calls
  :func:`~ui.db.update_strategy_status` with ``"APPROVED"``.
* Submission form (admin + analyst) — name, description, Python code;
  creates a :class:`~ui.db.Strategy` with ``status="PENDING"``.

Import discipline
-----------------
* Imports from ``ui.auth``, ``ui.db``, and ``ui.components`` only.
* No backtester engine imports.
* No cross-page imports.
"""

from __future__ import annotations

import streamlit as st

from ui.auth import get_current_user, require_role
from ui.db import create_strategy, get_db, list_strategies, list_users, update_strategy_status

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def strategy_library_page() -> None:
    """
    Render the strategy library browsing and submission page.

    Role enforcement
    ----------------
    Accessible to ``"admin"``, ``"analyst"``, and ``"viewer"``.

    Filter bar
    ----------
    * **Search** — substring match on strategy name and description.
    * **Status** (admin only) — ``"All"``, ``"APPROVED"``, ``"PENDING"``,
      ``"DRAFT"``; non-admins always see only ``"APPROVED"`` strategies.

    Strategy cards
    --------------
    Each visible strategy is rendered in an ``st.expander`` containing:

    * Description text and Python code block.
    * **🚀 Run This** button → sets ``st.session_state["page"] = "new_run"``
      and calls ``st.rerun()``.
    * **✅ Approve** button (admin + PENDING only) → calls
      :func:`~ui.db.update_strategy_status` with ``"APPROVED"``.

    Submission form
    ---------------
    Shown to ``"admin"`` and ``"analyst"`` at the bottom of the page.
    Requires a non-empty name and non-empty code; description is optional.
    On valid submit: calls :func:`~ui.db.create_strategy` with
    ``status="PENDING"`` and calls ``st.rerun()``.
    """
    require_role(["admin", "analyst", "viewer"])
    user = get_current_user()
    assert user is not None

    st.title("📚 Strategy Library")

    # ── Fetch data ─────────────────────────────────────────────────────────
    with get_db() as db:
        all_strategies = list_strategies(db)
        all_users = list_users(db)

    user_name_by_id: dict[int, str] = {u.id: u.name for u in all_users}

    # ── Filter bar ─────────────────────────────────────────────────────────
    search_q: str = st.text_input(
        "Search by name or description",
        value="",
        key="sl_search",
    )

    if user["role"] == "admin":
        status_filter: str = st.selectbox(
            "Filter by status",
            options=["All", "APPROVED", "PENDING", "DRAFT"],
            key="sl_status",
        )
    else:
        status_filter = "APPROVED"

    # ── Apply filters (client-side) ────────────────────────────────────────
    visible = all_strategies
    if status_filter != "All":
        visible = [s for s in visible if s.status == status_filter]
    sq = search_q.strip().lower()
    if sq:
        visible = [
            s for s in visible
            if sq in s.name.lower() or sq in s.description.lower()
        ]

    st.caption(
        f"Showing {len(visible)} strateg{'y' if len(visible) == 1 else 'ies'}"
    )

    # ── Strategy cards ─────────────────────────────────────────────────────
    if not visible:
        st.info("No strategies found.")
    else:
        for strategy in visible:
            author_name = user_name_by_id.get(
                strategy.author_id, f"User {strategy.author_id}"
            )
            label = (
                f"**{strategy.name}** — "
                f"by {author_name} ({strategy.status})"
            )
            with st.expander(label):
                st.markdown(strategy.description)
                st.code(strategy.code, language="python")

                act_col, adm_col = st.columns([3, 1])
                with act_col:
                    if st.button(
                        "🚀 Run This",
                        key=f"sl_run_{strategy.id}",
                    ):
                        st.session_state["page"] = "new_run"
                        st.rerun()

                if user["role"] == "admin" and strategy.status == "PENDING":
                    with adm_col:
                        if st.button(
                            "✅ Approve",
                            key=f"sl_approve_{strategy.id}",
                        ):
                            with get_db() as db:
                                update_strategy_status(
                                    db, strategy.id, "APPROVED"
                                )
                            st.success(f"'{strategy.name}' approved.")
                            st.rerun()

    # ── Submission form (admin + analyst only) ─────────────────────────────
    st.markdown("---")
    if user["role"] not in ("admin", "analyst"):
        return

    st.subheader("📤 Submit New Strategy")
    with st.form(key="sl_submit_form"):
        name = st.text_input("Strategy name", key="sl_name")
        description = st.text_area("Description (optional)", key="sl_description")
        code = st.text_area("Python code", key="sl_code")
        submitted = st.form_submit_button("Submit for Review")

    if not submitted:
        return

    if not name.strip():
        st.error("Strategy name is required.")
        return

    if not code.strip():
        st.error("Strategy code is required.")
        return

    with get_db() as db:
        create_strategy(
            db,
            name=name.strip(),
            description=description.strip(),
            code=code.strip(),
            author_id=user["user_id"],
        )
    st.success(f"Strategy '{name.strip()}' submitted for review!")
    st.rerun()
