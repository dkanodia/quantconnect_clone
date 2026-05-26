"""
Tests for ui/pages/notifications.py.

All Streamlit and DB calls are mocked — no live runtime or database needed.

Covers
------
empty inbox
    st.info shown when user has no notifications

notification display
    unread count shown in caption
    type badge HTML appears in rendered markdown
    human-readable message appears for RUN_DONE notification
    read notification does not show 🔵 indicator
    unread notification shows 🔵 indicator
    notifications are rendered in newest-first order

mark all read
    Mark-all button shown only when unread notifications exist
    Mark-all button NOT shown when all notifications are read
    clicking Mark-all calls mark_all_notifications_read
    rerun called after mark-all

individual mark read
    Mark-read button shown for unread notification
    Mark-read button NOT shown for already-read notification
    clicking Mark-read calls mark_notification_read with correct id
    rerun called after individual mark-read

open-run navigation
    Open button shown for RUN_DONE notification with run_id
    Open button shown for RUN_FAILED notification with run_id
    Open button shown for COMMENT notification with run_id
    Open button NOT shown for STRATEGY_APPROVED notification
    Open button NOT shown when run_id absent from payload
    clicking Open sets selected_run_id and navigates to run_detail
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from ui.pages.notifications import notifications_page


# ===========================================================================
# Shared test data
# ===========================================================================

_VIEWER_USER = {
    "user_id": 3,
    "name": "Carol Viewer",
    "email": "carol@example.com",
    "role": "viewer",
    "avatar_initials": "CV",
}


def _make_notif(
    notif_id: int = 1,
    notif_type: str = "RUN_DONE",
    read: bool = False,
    payload: dict | None = None,
    created_at: datetime | None = None,
) -> MagicMock:
    n = MagicMock()
    n.id = notif_id
    n.type = notif_type
    n.read = read
    n.payload = payload if payload is not None else {
        "run_id": "test-run-id",
        "strategy_name": "SMA",
    }
    n.created_at = created_at or datetime(2024, 6, 15, 12, 0)
    return n


# ===========================================================================
# Fixture
# ===========================================================================


@pytest.fixture
def mock_nt(monkeypatch):
    """
    Patch every external dependency of notifications_page() and return a
    control dict.
    """
    state: dict = {"page": "notifications", "selected_run_id": None}

    # ── Auth ───────────────────────────────────────────────────────────────
    monkeypatch.setattr("ui.pages.notifications.require_role", MagicMock())
    monkeypatch.setattr(
        "ui.pages.notifications.get_current_user",
        MagicMock(return_value=_VIEWER_USER),
    )

    # ── DB ─────────────────────────────────────────────────────────────────
    mock_db = MagicMock()

    @contextmanager
    def _mock_get_db():
        yield mock_db

    monkeypatch.setattr("ui.pages.notifications.get_db", _mock_get_db)

    mock_notifs: list = []
    mock_list_notifs = MagicMock(return_value=mock_notifs)
    monkeypatch.setattr(
        "ui.pages.notifications.list_notifications", mock_list_notifs
    )

    mock_mark_all = MagicMock()
    monkeypatch.setattr(
        "ui.pages.notifications.mark_all_notifications_read", mock_mark_all
    )

    mock_mark_one = MagicMock()
    monkeypatch.setattr(
        "ui.pages.notifications.mark_notification_read", mock_mark_one
    )

    # ── Streamlit mocks ────────────────────────────────────────────────────
    def _make_col() -> MagicMock:
        col = MagicMock()
        col.__enter__ = MagicMock(return_value=col)
        col.__exit__ = MagicMock(return_value=False)
        return col

    def _cols_side_effect(spec) -> list:
        n = len(spec) if isinstance(spec, (list, tuple)) else int(spec)
        return [_make_col() for _ in range(n)]

    mock_container_ctx = MagicMock()
    mock_container_ctx.__enter__ = MagicMock(return_value=mock_container_ctx)
    mock_container_ctx.__exit__ = MagicMock(return_value=False)

    st_mocks = {
        "title": MagicMock(),
        "caption": MagicMock(),
        "info": MagicMock(),
        "warning": MagicMock(),
        "error": MagicMock(),
        "success": MagicMock(),
        "markdown": MagicMock(),
        "columns": MagicMock(side_effect=_cols_side_effect),
        "container": MagicMock(return_value=mock_container_ctx),
        "button": MagicMock(return_value=False),
        "rerun": MagicMock(),
    }
    for name, mock in st_mocks.items():
        monkeypatch.setattr(f"ui.pages.notifications.st.{name}", mock)

    monkeypatch.setattr("ui.pages.notifications.st.session_state", state)

    return {
        "state": state,
        "notifs": mock_notifs,
        "list_notifs": mock_list_notifs,
        "mark_all": mock_mark_all,
        "mark_one": mock_mark_one,
        "st": st_mocks,
    }


# ===========================================================================
# Tests — empty inbox
# ===========================================================================


class TestEmptyInbox:
    def test_info_shown_when_no_notifications(self, mock_nt: dict) -> None:
        notifications_page()
        mock_nt["st"]["info"].assert_called_once()

    def test_list_notifications_called_for_current_user(
        self, mock_nt: dict
    ) -> None:
        notifications_page()
        call = mock_nt["list_notifs"].call_args
        # list_notifications(db, recipient_id=3)
        args = call[0]
        assert args[1] == _VIEWER_USER["user_id"]


# ===========================================================================
# Tests — notification display
# ===========================================================================


class TestNotificationDisplay:
    def test_unread_count_in_caption(self, mock_nt: dict) -> None:
        mock_nt["notifs"].extend([
            _make_notif(1, read=False),
            _make_notif(2, read=True),
        ])
        notifications_page()
        caption_texts = [
            str(c[0][0]) for c in mock_nt["st"]["caption"].call_args_list
        ]
        assert any("1 unread" in t for t in caption_texts)

    def test_type_badge_appears_in_markdown(self, mock_nt: dict) -> None:
        mock_nt["notifs"].append(_make_notif(notif_type="RUN_DONE"))
        notifications_page()
        md_texts = [
            str(c[0][0]) for c in mock_nt["st"]["markdown"].call_args_list
        ]
        assert any("RUN_DONE" in t for t in md_texts)

    def test_unread_indicator_present_for_unread(
        self, mock_nt: dict
    ) -> None:
        mock_nt["notifs"].append(_make_notif(read=False))
        notifications_page()
        md_texts = [
            str(c[0][0]) for c in mock_nt["st"]["markdown"].call_args_list
        ]
        assert any("🔵" in t for t in md_texts)

    def test_no_unread_indicator_for_read_notification(
        self, mock_nt: dict
    ) -> None:
        mock_nt["notifs"].append(_make_notif(read=True))
        notifications_page()
        md_texts = [
            str(c[0][0]) for c in mock_nt["st"]["markdown"].call_args_list
        ]
        assert not any("🔵" in t for t in md_texts)

    def test_human_readable_message_for_run_done(
        self, mock_nt: dict
    ) -> None:
        mock_nt["notifs"].append(
            _make_notif(notif_type="RUN_DONE", payload={"strategy_name": "SMA", "run_id": "r1"})
        )
        notifications_page()
        md_texts = [
            str(c[0][0]) for c in mock_nt["st"]["markdown"].call_args_list
        ]
        assert any("SMA" in t and "completed" in t for t in md_texts)

    def test_notifications_rendered_newest_first(
        self, mock_nt: dict
    ) -> None:
        older = _make_notif(1, "RUN_DONE", created_at=datetime(2024, 1, 1, 0, 0))
        newer = _make_notif(2, "COMMENT", created_at=datetime(2024, 6, 1, 0, 0))
        # Add older first in the list
        mock_nt["notifs"].extend([older, newer])
        notifications_page()
        md_texts = [
            str(c[0][0]) for c in mock_nt["st"]["markdown"].call_args_list
        ]
        comment_pos = next(
            (i for i, t in enumerate(md_texts) if "COMMENT" in t), None
        )
        run_done_pos = next(
            (i for i, t in enumerate(md_texts) if "RUN_DONE" in t), None
        )
        assert comment_pos is not None and run_done_pos is not None
        assert comment_pos < run_done_pos


# ===========================================================================
# Tests — mark all read
# ===========================================================================


class TestMarkAllRead:
    def test_mark_all_button_shown_when_unread_exist(
        self, mock_nt: dict
    ) -> None:
        mock_nt["notifs"].append(_make_notif(read=False))
        notifications_page()
        button_keys = [
            c.kwargs.get("key") for c in mock_nt["st"]["button"].call_args_list
        ]
        assert "nt_mark_all" in button_keys

    def test_mark_all_button_not_shown_when_all_read(
        self, mock_nt: dict
    ) -> None:
        mock_nt["notifs"].append(_make_notif(read=True))
        notifications_page()
        button_keys = [
            c.kwargs.get("key") for c in mock_nt["st"]["button"].call_args_list
        ]
        assert "nt_mark_all" not in button_keys

    def test_clicking_mark_all_calls_mark_all_notifications_read(
        self, mock_nt: dict
    ) -> None:
        mock_nt["notifs"].append(_make_notif(read=False))

        def _btn(*args, key: str = "", **kwargs):
            return key == "nt_mark_all"

        mock_nt["st"]["button"].side_effect = _btn
        notifications_page()
        mock_nt["mark_all"].assert_called_once()
        args = mock_nt["mark_all"].call_args[0]
        assert args[1] == _VIEWER_USER["user_id"]

    def test_rerun_called_after_mark_all(self, mock_nt: dict) -> None:
        mock_nt["notifs"].append(_make_notif(read=False))

        def _btn(*args, key: str = "", **kwargs):
            return key == "nt_mark_all"

        mock_nt["st"]["button"].side_effect = _btn
        notifications_page()
        mock_nt["st"]["rerun"].assert_called()


# ===========================================================================
# Tests — individual mark read
# ===========================================================================


class TestIndividualMarkRead:
    def test_mark_read_button_shown_for_unread(self, mock_nt: dict) -> None:
        notif = _make_notif(notif_id=7, read=False)
        mock_nt["notifs"].append(notif)
        notifications_page()
        button_keys = [
            c.kwargs.get("key") for c in mock_nt["st"]["button"].call_args_list
        ]
        assert "nt_read_7" in button_keys

    def test_mark_read_button_not_shown_for_read(
        self, mock_nt: dict
    ) -> None:
        notif = _make_notif(notif_id=8, read=True)
        mock_nt["notifs"].append(notif)
        notifications_page()
        button_keys = [
            c.kwargs.get("key") for c in mock_nt["st"]["button"].call_args_list
        ]
        assert "nt_read_8" not in button_keys

    def test_clicking_mark_read_calls_mark_notification_read(
        self, mock_nt: dict
    ) -> None:
        notif = _make_notif(notif_id=42, read=False)
        mock_nt["notifs"].append(notif)

        def _btn(*args, key: str = "", **kwargs):
            return key == "nt_read_42"

        mock_nt["st"]["button"].side_effect = _btn
        notifications_page()
        mock_nt["mark_one"].assert_called_once()
        args = mock_nt["mark_one"].call_args[0]
        assert args[1] == 42

    def test_rerun_called_after_individual_mark_read(
        self, mock_nt: dict
    ) -> None:
        notif = _make_notif(notif_id=55, read=False)
        mock_nt["notifs"].append(notif)

        def _btn(*args, key: str = "", **kwargs):
            return key == "nt_read_55"

        mock_nt["st"]["button"].side_effect = _btn
        notifications_page()
        mock_nt["st"]["rerun"].assert_called()


# ===========================================================================
# Tests — open-run navigation
# ===========================================================================


class TestOpenRunNavigation:
    def test_open_button_shown_for_run_done(self, mock_nt: dict) -> None:
        notif = _make_notif(
            notif_id=10, notif_type="RUN_DONE",
            payload={"run_id": "run-abc", "strategy_name": "SMA"},
        )
        mock_nt["notifs"].append(notif)
        notifications_page()
        button_keys = [
            c.kwargs.get("key") for c in mock_nt["st"]["button"].call_args_list
        ]
        assert "nt_open_10" in button_keys

    def test_open_button_shown_for_run_failed(self, mock_nt: dict) -> None:
        notif = _make_notif(
            notif_id=11, notif_type="RUN_FAILED",
            payload={"run_id": "run-abc", "strategy_name": "SMA"},
        )
        mock_nt["notifs"].append(notif)
        notifications_page()
        button_keys = [
            c.kwargs.get("key") for c in mock_nt["st"]["button"].call_args_list
        ]
        assert "nt_open_11" in button_keys

    def test_open_button_shown_for_comment(self, mock_nt: dict) -> None:
        notif = _make_notif(
            notif_id=12, notif_type="COMMENT",
            payload={"run_id": "run-abc", "commenter": "Bob"},
        )
        mock_nt["notifs"].append(notif)
        notifications_page()
        button_keys = [
            c.kwargs.get("key") for c in mock_nt["st"]["button"].call_args_list
        ]
        assert "nt_open_12" in button_keys

    def test_open_button_not_shown_for_strategy_approved(
        self, mock_nt: dict
    ) -> None:
        notif = _make_notif(
            notif_id=13, notif_type="STRATEGY_APPROVED",
            payload={"strategy_name": "SMA"},  # no run_id
        )
        mock_nt["notifs"].append(notif)
        notifications_page()
        button_keys = [
            c.kwargs.get("key") for c in mock_nt["st"]["button"].call_args_list
        ]
        assert "nt_open_13" not in button_keys

    def test_open_button_not_shown_when_no_run_id(
        self, mock_nt: dict
    ) -> None:
        notif = _make_notif(
            notif_id=14, notif_type="RUN_DONE",
            payload={"strategy_name": "SMA"},  # no run_id
        )
        mock_nt["notifs"].append(notif)
        notifications_page()
        button_keys = [
            c.kwargs.get("key") for c in mock_nt["st"]["button"].call_args_list
        ]
        assert "nt_open_14" not in button_keys

    def test_open_button_navigates_to_run_detail(
        self, mock_nt: dict
    ) -> None:
        notif = _make_notif(
            notif_id=20, notif_type="RUN_DONE",
            payload={"run_id": "my-run-id", "strategy_name": "SMA"},
        )
        mock_nt["notifs"].append(notif)

        def _btn(*args, key: str = "", **kwargs):
            return key == "nt_open_20"

        mock_nt["st"]["button"].side_effect = _btn
        notifications_page()
        assert mock_nt["state"]["selected_run_id"] == "my-run-id"
        assert mock_nt["state"]["page"] == "run_detail"
        mock_nt["st"]["rerun"].assert_called()
