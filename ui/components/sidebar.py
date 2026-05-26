"""
Sidebar component for the backtester Streamlit UI.

Exposes a single public function :func:`sidebar` that renders the full
left-hand navigation panel, user card, notification badge, and logout
button.  Called by :mod:`ui.app` after the authentication gate — never
called directly by individual page modules.

Import discipline
-----------------
* Only imports from :mod:`ui.auth`, :mod:`ui.db`, and ``streamlit``.
* No backtester engine imports.
* Opens its own :func:`~ui.db.get_db` session for the notification
  count — never reuses a session from the calling page.
"""

from __future__ import annotations

import streamlit as st

from ui.auth import get_current_user, logout
from ui.db import get_db, unread_count

# ---------------------------------------------------------------------------
# Navigation registry
# ---------------------------------------------------------------------------

NAV_ITEMS: list[dict] = [
    {
        "page": "dashboard",
        "label": "📊 Dashboard",
        "roles": ["admin", "analyst", "viewer"],
    },
    {
        "page": "new_run",
        "label": "▶️ New Run",
        "roles": ["admin", "analyst"],
    },
    {
        "page": "run_history",
        "label": "🕐 Run History",
        "roles": ["admin", "analyst", "viewer"],
    },
    {
        "page": "strategy_library",
        "label": "📚 Strategy Library",
        "roles": ["admin", "analyst", "viewer"],
    },
    {
        "page": "compare",
        "label": "⚖️ Compare Runs",
        "roles": ["admin", "analyst", "viewer"],
    },
    {
        "page": "notifications",
        "label": "🔔 Notifications",
        "roles": ["admin", "analyst", "viewer"],
    },
    {
        "page": "admin",
        "label": "⚙️ Admin Panel",
        "roles": ["admin"],
    },
]

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_SIDEBAR_CSS = """
<style>
.nav-item button {
    width: 100%;
    text-align: left;
    background: transparent;
    border: none;
    padding: 0.5rem 0.75rem;
    border-radius: 6px;
    font-size: 0.9rem;
}
.nav-active button {
    background: #f0f0f0;
    font-weight: 600;
}
.avatar-circle {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 36px;
    height: 36px;
    border-radius: 50%;
    background: #1a1a1a;
    color: white;
    font-size: 0.8rem;
    font-weight: 600;
}
</style>
"""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def sidebar() -> None:
    """
    Render the full sidebar: logo, nav items, notification badge, user card,
    and logout button.

    Must be called by the app router after the authentication gate — not
    directly by any page module.

    Role-based visibility
    ---------------------
    * ``"admin"``   — all 7 nav items.
    * ``"analyst"`` — 6 nav items (no Admin Panel).
    * ``"viewer"``  — 5 nav items (no New Run, no Admin Panel).

    Side effects
    ------------
    * Clicking a nav button writes to ``st.session_state["page"]`` and
      calls ``st.rerun()``.
    * Clicking the notification badge navigates to ``"notifications"``
      and calls ``st.rerun()``.
    * Clicking Logout calls :func:`~ui.auth.logout` and ``st.rerun()``.
    * Opens a short-lived DB session via :func:`~ui.db.get_db` to fetch
      the unread notification count.
    """
    user = get_current_user()
    if user is None:
        return  # should not reach here — called after auth gate

    # ── CSS injection ─────────────────────────────────────────────────────
    st.sidebar.markdown(_SIDEBAR_CSS, unsafe_allow_html=True)

    # ── 1. Logo + divider ─────────────────────────────────────────────────
    st.sidebar.markdown("## ⚡ Backtester")
    st.sidebar.markdown(
        "<hr style='margin: 0.25rem 0 0.75rem; border: none; "
        "border-top: 0.5px solid #e0e0e0;' />",
        unsafe_allow_html=True,
    )

    # ── 2. Nav items (filtered by role) ───────────────────────────────────
    current_page = st.session_state.get("page", "dashboard")
    role: str = user["role"]

    for item in NAV_ITEMS:
        if role not in item["roles"]:
            continue

        page_key: str = item["page"]
        label: str = item["label"]
        is_active: bool = page_key == current_page
        css_class = "nav-active" if is_active else "nav-item"

        st.sidebar.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)
        if st.sidebar.button(label, key=f"nav_{page_key}", use_container_width=True):
            st.session_state["page"] = page_key
            st.rerun()
        st.sidebar.markdown("</div>", unsafe_allow_html=True)

    # ── 3. Spacer ─────────────────────────────────────────────────────────
    st.sidebar.markdown("---")

    # ── 4. Notification badge (only when there are unread notifications) ──
    with get_db() as db:
        count: int = unread_count(db, user["user_id"])

    if count > 0:
        badge_label = f"🔔 {count} unread"
        badge_html = (
            f'<span style="background:#FFF3CD; color:#7A4F00; '
            f'padding:2px 8px; border-radius:4px; font-size:0.85rem;">'
            f'{badge_label}</span>'
        )
        st.sidebar.markdown(badge_html, unsafe_allow_html=True)
        if st.sidebar.button(badge_label, key="notif_badge"):
            st.session_state["page"] = "notifications"
            st.rerun()

    # ── 5. User card (avatar + name + role + logout) ──────────────────────
    avatar_html = (
        f'<div style="display:flex; align-items:center; gap:0.75rem; '
        f'padding:0.5rem 0;">'
        f'<div class="avatar-circle">{user["avatar_initials"]}</div>'
        f'<div>'
        f'<div style="font-weight:600; font-size:0.9rem;">{user["name"]}</div>'
        f'<div style="color:#666; font-size:0.8rem;">{user["role"]}</div>'
        f'</div>'
        f'</div>'
    )
    st.sidebar.markdown(avatar_html, unsafe_allow_html=True)

    if st.sidebar.button("🚪 Logout", key="logout_btn", use_container_width=True):
        logout()
        st.rerun()
