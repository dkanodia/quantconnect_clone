"""
Application entry point for the backtester Streamlit UI.

Run via::

    streamlit run ui/app.py

This module is the *sole* wiring point — it is the only file that imports
from all UI submodules.  Individual page modules import only from
``ui.auth``, ``ui.db``, and ``ui.components``.

Architecture
------------
* **Authentication gate** — unauthenticated users see only the login page.
* **Routing** — ``st.session_state["page"]`` drives which page is rendered.
  Any unknown key falls back to ``"dashboard"`` and triggers a rerun.
* **Sidebar** — rendered after the auth check on every authenticated load.

Session-state keys managed here
--------------------------------
``"page"``
    Current page key; defaults to ``"login"`` on first load.
``"selected_run_id"``
    UUID string of the run currently being viewed (``None`` if none).
``"compare_run_ids"``
    List of run UUIDs selected for side-by-side comparison.
``"run_history_filters"``
    Dict of active filter values on the Run History page.

Import discipline
-----------------
* This file is the only one that imports from all ``ui.*`` submodules.
* Pages import only from ``ui.auth``, ``ui.db``, and ``ui.components``.
* No backtester engine imports here.
"""

from __future__ import annotations

from typing import Callable

import streamlit as st

from ui.auth import is_authenticated
from ui.components.sidebar import sidebar
from ui.db import init_db
from ui.pages.admin import admin_page
from ui.pages.compare import compare_page
from ui.pages.dashboard import dashboard_page
from ui.pages.login import login_page
from ui.pages.new_run import new_run_page
from ui.pages.notifications import notifications_page
from ui.pages.run_detail import run_detail_page
from ui.pages.run_history import run_history_page
from ui.pages.strategy_library import strategy_library_page

# ---------------------------------------------------------------------------
# Global CSS — light fintech aesthetic, injected once per load
# ---------------------------------------------------------------------------

_GLOBAL_CSS: str = """
<style>
body { font-family: 'Inter', sans-serif; }
[data-testid="stSidebar"] {
    border-right: 0.5px solid #e0e0e0;
    background: #fafafa;
}
[data-testid="stAppViewContainer"] {
    background: #ffffff;
}
.metric-card {
    border: 0.5px solid #e0e0e0;
    border-radius: 6px;
    padding: 1rem;
    background: #ffffff;
}
.positive { color: #27500A; background: #EAF3DE; padding: 2px 6px; border-radius: 4px; }
.negative { color: #791F1F; background: #FCEBEB; padding: 2px 6px; border-radius: 4px; }
.badge-wfo { color: #0C447C; background: #E6F1FB; padding: 2px 6px; border-radius: 4px; }
.badge-pending { color: #7A4F00; background: #FFF3CD; padding: 2px 6px; border-radius: 4px; }
</style>
"""

# ---------------------------------------------------------------------------
# Stub page factory
# ---------------------------------------------------------------------------


def _stub_page(name: str) -> Callable[[], None]:
    """
    Return a zero-argument placeholder page callable for pages not yet built.

    The returned callable renders a ``st.info`` "coming soon" banner.  Use
    it in :data:`PAGES` to register a route before its full implementation
    exists.

    Parameters
    ----------
    name:
        Human-readable page name shown in the banner
        (e.g. ``"Admin Panel"``).

    Returns
    -------
    Callable[[], None]
        A page function compatible with the :data:`PAGES` registry.

    Example
    -------
    ::

        PAGES["my_page"] = _stub_page("My Page")
    """
    def _page() -> None:
        st.info(f"🚧 {name} — coming soon")

    _page.__name__ = f"stub_{name.lower().replace(' ', '_')}"
    return _page


# ---------------------------------------------------------------------------
# Page registry
# ---------------------------------------------------------------------------

PAGES: dict[str, Callable[[], None]] = {
    "login":            login_page,
    "dashboard":        dashboard_page,
    "new_run":          new_run_page,
    "run_history":      run_history_page,
    "run_detail":       run_detail_page,
    "compare":          compare_page,
    "strategy_library": strategy_library_page,
    "notifications":    notifications_page,
    "admin":            admin_page,
}

# ---------------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Top-level Streamlit entry point — called once per script re-run.

    Steps performed on every load
    ------------------------------
    1. Configure the Streamlit page (title, icon, layout, sidebar state).
    2. Inject global CSS.
    3. Initialise missing session-state keys to their defaults.
    4. Call :func:`~ui.db.init_db` (idempotent — creates tables if absent).
    5. **Auth gate**:

       * Not authenticated → render :func:`~ui.pages.login.login_page`
         then call ``st.stop()``.  The sidebar is *not* rendered.
       * Authenticated → render :func:`~ui.components.sidebar.sidebar`,
         then route to the page function keyed by
         ``st.session_state["page"]``.

    6. **Unknown page guard** — if ``session_state["page"]`` is not in
       :data:`PAGES`, reset it to ``"dashboard"`` and call ``st.rerun()``.

    Session-state defaults
    ----------------------
    All four keys are set with ``setdefault`` so existing values are never
    overwritten::

        "page"                → "login"
        "selected_run_id"     → None
        "compare_run_ids"     → []
        "run_history_filters" → {}
    """
    # ── 1. Page configuration ──────────────────────────────────────────────
    st.set_page_config(
        page_title="Backtester",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── 2. Global CSS ──────────────────────────────────────────────────────
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)

    # ── 3. Session-state defaults ──────────────────────────────────────────
    st.session_state.setdefault("page", "login")
    st.session_state.setdefault("selected_run_id", None)
    st.session_state.setdefault("compare_run_ids", [])
    st.session_state.setdefault("run_history_filters", {})

    # ── 4. Database initialisation (idempotent) ────────────────────────────
    init_db()

    # ── 5. Authentication gate ─────────────────────────────────────────────
    if not is_authenticated():
        login_page()
        st.stop()
        return

    # ── 6. Sidebar + page routing ──────────────────────────────────────────
    sidebar()

    page_key: str = st.session_state["page"]
    if page_key not in PAGES:
        st.session_state["page"] = "dashboard"
        st.rerun()
        return

    PAGES[page_key]()


# Streamlit executes this script as __main__ on every user interaction.
if __name__ == "__main__":
    main()
