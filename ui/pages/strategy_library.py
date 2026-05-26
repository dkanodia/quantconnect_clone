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
* Submission editor (admin + analyst) — VSCode-style two-column layout with
  an Ace code editor (monokai theme, Python syntax highlighting, line numbers).

Import discipline
-----------------
* Imports from ``ui.auth``, ``ui.db``, ``ui.components``, and
  ``streamlit_ace`` only.
* No backtester engine imports.
* No cross-page imports.
"""

from __future__ import annotations

import streamlit as st
from streamlit_ace import st_ace

from ui.auth import get_current_user, require_role
from ui.db import (
    create_strategy,
    get_db,
    list_strategies,
    list_users,
    update_strategy_status,
)

# ---------------------------------------------------------------------------
# Default starter template shown in the editor for new strategies
# ---------------------------------------------------------------------------

_DEFAULT_CODE_TEMPLATE = """\
from backtester.strategy import Strategy


class MyStrategy(Strategy):
    \"\"\"Replace this docstring with a short description of your strategy.\"\"\"

    def __init__(self):
        super().__init__()
        # Declare hyper-parameters here, e.g.:
        # self.fast_window = 20
        # self.slow_window = 50

    def on_bar(self, bar):
        \"\"\"Called once per price bar.  Implement your signal logic here.\"\"\"
        pass
"""

# ---------------------------------------------------------------------------
# CSS — VSCode-like editor chrome injected only for the submission section
# ---------------------------------------------------------------------------

_EDITOR_CSS = """
<style>
/* Editor tab bar — mirrors VS Code's tab strip */
.vsc-tab-bar {
    display: flex;
    align-items: stretch;
    background: #1e1e1e;
    border-radius: 6px 6px 0 0;
    border: 1px solid #3c3c3c;
    border-bottom: none;
    height: 34px;
    overflow: hidden;
}
.vsc-tab {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 0 16px;
    background: #2d2d2d;
    color: #cccccc;
    font-size: 0.78rem;
    font-family: 'SF Mono', 'Menlo', 'Monaco', 'Consolas', monospace;
    border-right: 1px solid #3c3c3c;
    border-bottom: 2px solid #4f6ef7;
    white-space: nowrap;
}
.vsc-tab-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #4f6ef7;
    flex-shrink: 0;
}
.vsc-tab-spacer {
    flex: 1;
}
/* Drop the gap below the tab bar so the editor butts up against it */
.vsc-editor-wrap > div:first-child { margin-top: 0 !important; }

/* Metadata panel submit hint */
.submit-hint {
    font-size: 0.76rem;
    opacity: 0.55;
    line-height: 1.5;
    margin: 0;
}
.submit-hint strong { opacity: 0.9; }
</style>
"""

# Status emoji for strategy cards
_STATUS_ICON: dict[str, str] = {
    "APPROVED": "🟢",
    "PENDING":  "🟡",
    "DRAFT":    "⚪",
}


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

    * Description caption and Python code block.
    * **🚀 Run This** button → sets ``st.session_state["page"] = "new_run"``
      and calls ``st.rerun()``.
    * **✅ Approve** button (admin + PENDING only) → calls
      :func:`~ui.db.update_strategy_status` with ``"APPROVED"``.

    Submission editor
    -----------------
    Shown to ``"admin"`` and ``"analyst"`` at the bottom of the page.
    Two-column layout: left panel = metadata (name, description, submit
    button); right panel = VSCode-style Ace editor with monokai theme,
    Python syntax highlighting, and line numbers.
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
    if user["role"] == "admin":
        _f_col, _s_col = st.columns([3, 1])
        with _f_col:
            search_q: str = st.text_input(
                "Search",
                placeholder="Filter by name or description…",
                value="",
                key="sl_search",
                label_visibility="collapsed",
            )
        with _s_col:
            status_filter: str = st.selectbox(
                "Filter by status",
                options=["All", "APPROVED", "PENDING", "DRAFT"],
                key="sl_status",
                label_visibility="collapsed",
            )
    else:
        search_q = st.text_input(
            "Search",
            placeholder="Filter by name or description…",
            value="",
            key="sl_search",
            label_visibility="collapsed",
        )
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
        f"{len(visible)} strateg{'y' if len(visible) == 1 else 'ies'}"
    )

    # ── Strategy cards ─────────────────────────────────────────────────────
    if not visible:
        st.info("No strategies match your search.")
    else:
        for strategy in visible:
            author_name = user_name_by_id.get(
                strategy.author_id, f"User {strategy.author_id}"
            )
            icon = _STATUS_ICON.get(strategy.status, "⚪")
            label = (
                f"{icon} **{strategy.name}** "
                f"— {author_name} · {strategy.status}"
            )
            with st.expander(label):
                if strategy.description:
                    st.caption(strategy.description)
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

    # ── Submission editor (admin + analyst only) ───────────────────────────
    st.markdown("---")
    if user["role"] not in ("admin", "analyst"):
        return

    st.markdown(_EDITOR_CSS, unsafe_allow_html=True)
    st.subheader("📝 Submit New Strategy")
    st.caption(
        "Write your strategy below. It will be **PENDING** until an admin approves it."
    )

    meta_col, editor_col = st.columns([2, 3])

    # ── Left: metadata panel ───────────────────────────────────────────────
    with meta_col:
        with st.container(border=True):
            name: str = st.text_input(
                "Strategy name",
                placeholder="e.g. SMA Crossover",
                key="sl_name",
            )
            description: str = st.text_area(
                "Description",
                placeholder=(
                    "What does this strategy do?\n"
                    "What instruments or timeframes does it trade?"
                ),
                key="sl_description",
                height=120,
            )
            st.divider()
            st.markdown(
                '<p class="submit-hint">'
                "Submitted strategies are <strong>PENDING</strong> review. "
                "An admin will approve or reject before it appears to the team."
                "</p>",
                unsafe_allow_html=True,
            )
            submitted: bool = st.button(
                "Submit for Review",
                key="sl_submit",
                use_container_width=True,
                type="primary",
            )

    # ── Right: VSCode-style Ace editor ────────────────────────────────────
    with editor_col:
        st.markdown(
            '<div class="vsc-tab-bar">'
            '  <div class="vsc-tab">'
            '    <span class="vsc-tab-dot"></span>'
            "    strategy.py"
            "  </div>"
            '  <div class="vsc-tab-spacer"></div>'
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown('<div class="vsc-editor-wrap">', unsafe_allow_html=True)
        code: str = st_ace(
            value=_DEFAULT_CODE_TEMPLATE,
            language="python",
            theme="monokai",
            height=340,
            font_size=13,
            tab_size=4,
            show_gutter=True,
            show_print_margin=False,
            wrap=False,
            key="sl_code_editor",
            auto_update=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Validation + persistence ───────────────────────────────────────────
    if not submitted:
        return

    if not name.strip():
        st.error("Strategy name is required.")
        return

    if not code or not code.strip():
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
