"""
Notifications page — inbox of in-app alerts for the current user.

Features
--------
* Lists all notifications in reverse-chronological order.
* Unread indicator + count; "Mark all read" button clears the badge.
* Per-notification "Mark read" button for individual notifications.
* "Open →" button for run-linked notifications (RUN_DONE, RUN_FAILED,
  COMMENT) navigates to the run detail page.
* Type badge: RUN_DONE, RUN_FAILED, COMMENT, STRATEGY_APPROVED.

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
    list_notifications,
    mark_all_notifications_read,
    mark_notification_read,
)

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

_TYPE_COLORS: dict[str, tuple[str, str]] = {
    "RUN_DONE":          ("#27500A", "#EAF3DE"),
    "RUN_FAILED":        ("#791F1F", "#FCEBEB"),
    "COMMENT":           ("#0C447C", "#E6F1FB"),
    "STRATEGY_APPROVED": ("#7A4F00", "#FFF3CD"),
}

_RUN_TYPES = frozenset({"RUN_DONE", "RUN_FAILED", "COMMENT"})


def _type_badge(notif_type: str) -> str:
    """Return an inline-HTML badge for a notification type string."""
    color, bg = _TYPE_COLORS.get(notif_type, ("#6b6b6b", "#f0f0f0"))
    return (
        f'<span style="color:{color}; background:{bg}; padding:2px 6px; '
        f'border-radius:4px; font-size:0.75rem; font-weight:500;">'
        f"{notif_type}</span>"
    )


def _notif_message(notif_type: str, payload: dict) -> str:
    """Build a human-readable message from notification type and payload."""
    if notif_type == "RUN_DONE":
        name = payload.get("strategy_name", "Unknown")
        return f"Your run **{name}** completed successfully."
    if notif_type == "RUN_FAILED":
        name = payload.get("strategy_name", "Unknown")
        return f"Your run **{name}** failed."
    if notif_type == "COMMENT":
        who = payload.get("commenter", "Someone")
        return f"**{who}** commented on your run."
    if notif_type == "STRATEGY_APPROVED":
        name = payload.get("strategy_name", "Unknown")
        return f"Your strategy **{name}** was approved."
    return notif_type


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def notifications_page() -> None:
    """
    Render the notification inbox for the current user.

    Role enforcement
    ----------------
    Accessible to ``"admin"``, ``"analyst"``, and ``"viewer"``.

    Layout
    ------
    * Caption showing the unread count.
    * **Mark all read** button (shown only when unread > 0).
    * One bordered card per notification, newest first, containing:

      - Type badge, read indicator (🔵), and human-readable message.
      - Timestamp caption.
      - **Mark read** button (unread notifications only).
      - **Open →** button for run-linked types (RUN_DONE, RUN_FAILED,
        COMMENT); navigates to ``run_detail`` and calls ``st.rerun()``.
    """
    require_role(["admin", "analyst", "viewer"])
    user = get_current_user()
    assert user is not None

    st.title("🔔 Notifications")

    # ── Fetch ──────────────────────────────────────────────────────────────
    with get_db() as db:
        all_notifs = list_notifications(db, user["user_id"])

    if not all_notifs:
        st.info("No notifications yet.")
        return

    unread = [n for n in all_notifs if not n.read]
    st.caption(f"{len(unread)} unread")

    # ── Mark all read ──────────────────────────────────────────────────────
    if unread:
        if st.button("✅ Mark all read", key="nt_mark_all"):
            with get_db() as db:
                mark_all_notifications_read(db, user["user_id"])
            st.rerun()

    # ── Notification cards (newest first) ──────────────────────────────────
    sorted_notifs = sorted(all_notifs, key=lambda n: n.created_at, reverse=True)

    for notif in sorted_notifs:
        payload: dict = notif.payload or {}
        badge_html = _type_badge(notif.type)
        msg = _notif_message(notif.type, payload)
        indicator = "🔵 " if not notif.read else ""

        with st.container(border=True):
            left, right = st.columns([5, 1])

            with left:
                st.markdown(
                    f"{indicator}{badge_html} {msg}",
                    unsafe_allow_html=True,
                )
                st.caption(notif.created_at.strftime("%Y-%m-%d %H:%M UTC"))

            with right:
                if not notif.read:
                    if st.button("Mark read", key=f"nt_read_{notif.id}"):
                        with get_db() as db:
                            mark_notification_read(db, notif.id)
                        st.rerun()

                run_id = payload.get("run_id")
                if run_id and notif.type in _RUN_TYPES:
                    if st.button("Open →", key=f"nt_open_{notif.id}"):
                        st.session_state["selected_run_id"] = run_id
                        st.session_state["page"] = "run_detail"
                        st.rerun()
