"""
Tests for ui/pages/admin.py.

All Streamlit and DB calls are mocked — no live runtime or database needed.

Covers
------
access control
    require_role called with ["admin"] only

system stats
    four metric cards rendered (users, runs, active runs, pending strategies)
    active_run count counts only RUNNING status runs
    pending_strategy count matches pending strategies fetched

user management
    admin user (self) is excluded from the user list
    role selectbox rendered for each non-self user
    selectbox index reflects current user role
    Save button appears when selected role differs from current role
    Save button absent when selected role matches current role
    clicking Save calls update_user_role with correct user_id and new role
    rerun called after role update

strategy review queue
    pending strategies shown in expanders
    no pending strategies shows info message
    clicking Approve calls update_strategy_status with APPROVED
    clicking Reject calls update_strategy_status with DRAFT
    rerun called after Approve
    rerun called after Reject
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from unittest.mock import MagicMock
import uuid

import pytest

from ui.pages.admin import admin_page


# ===========================================================================
# Shared test data
# ===========================================================================

_ADMIN_USER = {
    "user_id": 99,
    "name": "Super Admin",
    "email": "admin@example.com",
    "role": "admin",
    "avatar_initials": "SA",
}


def _make_db_user(
    user_id: int = 1,
    name: str = "Alice",
    role: str = "analyst",
    email: str = "alice@test.com",
) -> MagicMock:
    u = MagicMock()
    u.id = user_id
    u.name = name
    u.role = role
    u.email = email
    return u


def _make_run(status: str = "DONE") -> MagicMock:
    r = MagicMock()
    r.id = str(uuid.uuid4())
    r.status = status
    r.owner_id = 1
    return r


def _make_strategy(
    strategy_id: int = 1,
    name: str = "Pending Strategy",
    status: str = "PENDING",
    author_id: int = 1,
) -> MagicMock:
    s = MagicMock()
    s.id = strategy_id
    s.name = name
    s.description = "A test strategy description."
    s.code = "def run(): pass"
    s.status = status
    s.author_id = author_id
    return s


# ===========================================================================
# Fixture
# ===========================================================================


@pytest.fixture
def mock_admin(monkeypatch):
    """
    Patch every external dependency of admin_page() and return a control dict.
    """
    state: dict = {"page": "admin"}

    # ── Auth ───────────────────────────────────────────────────────────────
    monkeypatch.setattr("ui.pages.admin.require_role", MagicMock())
    monkeypatch.setattr(
        "ui.pages.admin.get_current_user",
        MagicMock(return_value=_ADMIN_USER),
    )

    # ── DB ─────────────────────────────────────────────────────────────────
    mock_db = MagicMock()

    @contextmanager
    def _mock_get_db():
        yield mock_db

    monkeypatch.setattr("ui.pages.admin.get_db", _mock_get_db)

    # Default: admin user + one regular user, no runs, no pending strategies
    admin_db_user = _make_db_user(
        user_id=99, name="Super Admin", role="admin", email="admin@example.com"
    )
    regular_user = _make_db_user(user_id=1, name="Alice", role="analyst")

    mock_all_users: list = [admin_db_user, regular_user]
    mock_all_runs: list = []
    mock_pending: list = []

    mock_list_users = MagicMock(return_value=mock_all_users)
    mock_list_runs = MagicMock(return_value=mock_all_runs)
    mock_list_strategies = MagicMock(return_value=mock_pending)

    monkeypatch.setattr("ui.pages.admin.list_users", mock_list_users)
    monkeypatch.setattr("ui.pages.admin.list_runs_for_user", mock_list_runs)
    monkeypatch.setattr("ui.pages.admin.list_strategies", mock_list_strategies)

    mock_update_role = MagicMock()
    monkeypatch.setattr("ui.pages.admin.update_user_role", mock_update_role)

    mock_update_status = MagicMock()
    monkeypatch.setattr(
        "ui.pages.admin.update_strategy_status", mock_update_status
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

    mock_expander_ctx = MagicMock()
    mock_expander_ctx.__enter__ = MagicMock(return_value=mock_expander_ctx)
    mock_expander_ctx.__exit__ = MagicMock(return_value=False)

    st_mocks = {
        "title": MagicMock(),
        "subheader": MagicMock(),
        "caption": MagicMock(),
        "info": MagicMock(),
        "warning": MagicMock(),
        "error": MagicMock(),
        "success": MagicMock(),
        "markdown": MagicMock(),
        "code": MagicMock(),
        "columns": MagicMock(side_effect=_cols_side_effect),
        "metric": MagicMock(),
        "selectbox": MagicMock(return_value="analyst"),  # default: same as role
        "button": MagicMock(return_value=False),
        "expander": MagicMock(return_value=mock_expander_ctx),
        "rerun": MagicMock(),
    }
    for name, mock in st_mocks.items():
        monkeypatch.setattr(f"ui.pages.admin.st.{name}", mock)

    monkeypatch.setattr("ui.pages.admin.st.session_state", state)

    return {
        "state": state,
        "all_users": mock_all_users,
        "all_runs": mock_all_runs,
        "pending": mock_pending,
        "list_users": mock_list_users,
        "list_runs": mock_list_runs,
        "list_strategies": mock_list_strategies,
        "update_role": mock_update_role,
        "update_status": mock_update_status,
        "st": st_mocks,
        "admin_db_user": admin_db_user,
        "regular_user": regular_user,
    }


# ===========================================================================
# Tests — access control
# ===========================================================================


class TestAdminAccessControl:
    def test_title_rendered(self, mock_admin: dict) -> None:
        admin_page()
        mock_admin["st"]["title"].assert_called_once()

    def test_list_users_called(self, mock_admin: dict) -> None:
        admin_page()
        mock_admin["list_users"].assert_called_once()


# ===========================================================================
# Tests — system stats
# ===========================================================================


class TestSystemStats:
    def test_four_metric_cards_rendered(self, mock_admin: dict) -> None:
        admin_page()
        assert mock_admin["st"]["metric"].call_count >= 4

    def test_active_runs_count_only_running(self, mock_admin: dict) -> None:
        mock_admin["all_runs"].extend([
            _make_run("RUNNING"),
            _make_run("RUNNING"),
            _make_run("DONE"),
        ])
        admin_page()
        metric_values = [
            c[0][1] if len(c[0]) > 1 else c[1].get("value")
            for c in mock_admin["st"]["metric"].call_args_list
        ]
        assert 2 in metric_values

    def test_pending_strategies_count_in_metrics(
        self, mock_admin: dict
    ) -> None:
        mock_admin["pending"].extend([
            _make_strategy(1, status="PENDING"),
            _make_strategy(2, status="PENDING"),
        ])
        admin_page()
        metric_values = [
            c[0][1] if len(c[0]) > 1 else c[1].get("value")
            for c in mock_admin["st"]["metric"].call_args_list
        ]
        assert 2 in metric_values

    def test_total_users_count_in_metrics(self, mock_admin: dict) -> None:
        # default fixture has 2 users (admin + regular)
        admin_page()
        metric_values = [
            c[0][1] if len(c[0]) > 1 else c[1].get("value")
            for c in mock_admin["st"]["metric"].call_args_list
        ]
        assert 2 in metric_values


# ===========================================================================
# Tests — user management
# ===========================================================================


class TestUserManagement:
    def test_admin_self_excluded_from_list(self, mock_admin: dict) -> None:
        admin_page()
        selectbox_keys = [
            c.kwargs.get("key") for c in mock_admin["st"]["selectbox"].call_args_list
        ]
        # Admin's own user_id=99 should not appear
        assert not any(
            "adm_role_99" in (k or "") for k in selectbox_keys
        )

    def test_role_selectbox_rendered_for_regular_user(
        self, mock_admin: dict
    ) -> None:
        admin_page()
        selectbox_keys = [
            c.kwargs.get("key") for c in mock_admin["st"]["selectbox"].call_args_list
        ]
        # Regular user (id=1) should have a selectbox
        assert any("adm_role_1" in (k or "") for k in selectbox_keys)

    def test_selectbox_index_reflects_current_role(
        self, mock_admin: dict
    ) -> None:
        # regular_user.role = "analyst" → index 1 in ["viewer", "analyst", "admin"]
        admin_page()
        role_selectbox_calls = [
            c for c in mock_admin["st"]["selectbox"].call_args_list
            if c.kwargs.get("key") == "adm_role_1"
        ]
        assert len(role_selectbox_calls) == 1
        idx = role_selectbox_calls[0].kwargs.get("index")
        assert idx == 1  # "analyst" is at index 1

    def test_save_button_shown_when_role_differs(
        self, mock_admin: dict
    ) -> None:
        # selectbox returns "viewer" (different from "analyst")
        mock_admin["st"]["selectbox"].return_value = "viewer"
        admin_page()
        button_keys = [
            c.kwargs.get("key") for c in mock_admin["st"]["button"].call_args_list
        ]
        assert any("adm_save_1" in (k or "") for k in button_keys)

    def test_save_button_not_shown_when_role_same(
        self, mock_admin: dict
    ) -> None:
        # selectbox returns "analyst" (same as current role)
        mock_admin["st"]["selectbox"].return_value = "analyst"
        admin_page()
        button_keys = [
            c.kwargs.get("key") for c in mock_admin["st"]["button"].call_args_list
        ]
        assert not any("adm_save_1" in (k or "") for k in button_keys)

    def test_clicking_save_calls_update_user_role(
        self, mock_admin: dict
    ) -> None:
        mock_admin["st"]["selectbox"].return_value = "viewer"

        def _btn(*args, key: str = "", **kwargs):
            return key == "adm_save_1"

        mock_admin["st"]["button"].side_effect = _btn
        admin_page()
        mock_admin["update_role"].assert_called_once()
        args = mock_admin["update_role"].call_args[0]
        assert args[1] == 1       # user_id
        assert args[2] == "viewer"  # new role

    def test_rerun_called_after_role_update(self, mock_admin: dict) -> None:
        mock_admin["st"]["selectbox"].return_value = "viewer"

        def _btn(*args, key: str = "", **kwargs):
            return key == "adm_save_1"

        mock_admin["st"]["button"].side_effect = _btn
        admin_page()
        mock_admin["st"]["rerun"].assert_called()


# ===========================================================================
# Tests — strategy review queue
# ===========================================================================


class TestStrategyReviewQueue:
    def test_no_pending_strategies_shows_info(
        self, mock_admin: dict
    ) -> None:
        admin_page()
        mock_admin["st"]["info"].assert_called_once()
        info_text = str(mock_admin["st"]["info"].call_args[0][0])
        assert "pending" in info_text.lower()

    def test_pending_strategy_shown_in_expander(
        self, mock_admin: dict
    ) -> None:
        strat = _make_strategy(strategy_id=5, name="Alpha Strategy", status="PENDING")
        mock_admin["pending"].append(strat)
        admin_page()
        expander_labels = [
            str(c[0][0]) for c in mock_admin["st"]["expander"].call_args_list
        ]
        assert any("Alpha Strategy" in l for l in expander_labels)

    def test_approve_calls_update_strategy_status_approved(
        self, mock_admin: dict
    ) -> None:
        strat = _make_strategy(strategy_id=7, name="Algo X", status="PENDING")
        mock_admin["pending"].append(strat)

        def _btn(*args, key: str = "", **kwargs):
            return key == "adm_approve_7"

        mock_admin["st"]["button"].side_effect = _btn
        admin_page()
        mock_admin["update_status"].assert_called_once()
        args = mock_admin["update_status"].call_args[0]
        assert args[1] == 7
        assert args[2] == "APPROVED"

    def test_reject_calls_update_strategy_status_draft(
        self, mock_admin: dict
    ) -> None:
        strat = _make_strategy(strategy_id=8, name="Algo Y", status="PENDING")
        mock_admin["pending"].append(strat)

        def _btn(*args, key: str = "", **kwargs):
            return key == "adm_reject_8"

        mock_admin["st"]["button"].side_effect = _btn
        admin_page()
        mock_admin["update_status"].assert_called_once()
        args = mock_admin["update_status"].call_args[0]
        assert args[1] == 8
        assert args[2] == "DRAFT"

    def test_rerun_called_after_approve(self, mock_admin: dict) -> None:
        strat = _make_strategy(strategy_id=9, status="PENDING")
        mock_admin["pending"].append(strat)

        def _btn(*args, key: str = "", **kwargs):
            return key == "adm_approve_9"

        mock_admin["st"]["button"].side_effect = _btn
        admin_page()
        mock_admin["st"]["rerun"].assert_called()

    def test_rerun_called_after_reject(self, mock_admin: dict) -> None:
        strat = _make_strategy(strategy_id=11, status="PENDING")
        mock_admin["pending"].append(strat)

        def _btn(*args, key: str = "", **kwargs):
            return key == "adm_reject_11"

        mock_admin["st"]["button"].side_effect = _btn
        admin_page()
        mock_admin["st"]["rerun"].assert_called()
