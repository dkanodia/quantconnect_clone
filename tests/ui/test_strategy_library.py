"""
Tests for ui/pages/strategy_library.py.

All Streamlit and DB calls are mocked — no live runtime or database needed.

Covers
------
display
    approved strategies rendered as expanders
    no strategies shows info message
    strategy name and author appear in expander label

filtering
    search narrows visible strategies by name
    search narrows visible strategies by description
    non-admin always sees only APPROVED strategies (status filter hidden)
    admin status filter PENDING shows only pending strategies

run-this navigation
    clicking "Run This" sets page = "new_run" and calls rerun

admin approval
    Approve button shown only for admin + PENDING strategy
    clicking Approve calls update_strategy_status with APPROVED

submission form
    form shown for analyst
    form shown for admin
    form hidden for viewer
    empty name shows error
    empty code shows error
    valid submission calls create_strategy with correct args
    rerun called after successful submission
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

from ui.pages.strategy_library import strategy_library_page


# ===========================================================================
# Shared test data
# ===========================================================================

_ANALYST_USER = {
    "user_id": 1,
    "name": "Alice Analyst",
    "email": "alice@example.com",
    "role": "analyst",
    "avatar_initials": "AA",
}

_ADMIN_USER = {
    "user_id": 2,
    "name": "Bob Admin",
    "email": "bob@example.com",
    "role": "admin",
    "avatar_initials": "BA",
}

_VIEWER_USER = {
    "user_id": 3,
    "name": "Carol Viewer",
    "email": "carol@example.com",
    "role": "viewer",
    "avatar_initials": "CV",
}


def _make_strategy(
    strategy_id: int = 1,
    name: str = "SMA Crossover",
    description: str = "A simple moving average strategy.",
    code: str = "def run(): pass",
    author_id: int = 1,
    status: str = "APPROVED",
) -> MagicMock:
    s = MagicMock()
    s.id = strategy_id
    s.name = name
    s.description = description
    s.code = code
    s.author_id = author_id
    s.status = status
    return s


def _make_db_user(user_id: int = 1, name: str = "Alice Analyst") -> MagicMock:
    u = MagicMock()
    u.id = user_id
    u.name = name
    return u


# ===========================================================================
# Fixture
# ===========================================================================


@pytest.fixture
def mock_sl(monkeypatch):
    """
    Patch every external dependency of strategy_library_page() and return a
    control dict.
    """
    state: dict = {"page": "strategy_library"}

    # ── Auth ───────────────────────────────────────────────────────────────
    monkeypatch.setattr("ui.pages.strategy_library.require_role", MagicMock())
    monkeypatch.setattr(
        "ui.pages.strategy_library.get_current_user",
        MagicMock(return_value=_ANALYST_USER),
    )

    # ── DB ─────────────────────────────────────────────────────────────────
    mock_db = MagicMock()

    @contextmanager
    def _mock_get_db():
        yield mock_db

    monkeypatch.setattr("ui.pages.strategy_library.get_db", _mock_get_db)

    mock_strategies: list = []
    mock_list_strategies = MagicMock(return_value=mock_strategies)
    monkeypatch.setattr(
        "ui.pages.strategy_library.list_strategies", mock_list_strategies
    )

    mock_db_user = _make_db_user(user_id=1, name="Alice Analyst")
    mock_list_users = MagicMock(return_value=[mock_db_user])
    monkeypatch.setattr(
        "ui.pages.strategy_library.list_users", mock_list_users
    )

    mock_create_strategy = MagicMock()
    monkeypatch.setattr(
        "ui.pages.strategy_library.create_strategy", mock_create_strategy
    )

    mock_update_status = MagicMock()
    monkeypatch.setattr(
        "ui.pages.strategy_library.update_strategy_status", mock_update_status
    )

    # ── Streamlit mocks ────────────────────────────────────────────────────
    mock_expander_ctx = MagicMock()
    mock_expander_ctx.__enter__ = MagicMock(return_value=mock_expander_ctx)
    mock_expander_ctx.__exit__ = MagicMock(return_value=False)

    mock_form_ctx = MagicMock()
    mock_form_ctx.__enter__ = MagicMock(return_value=mock_form_ctx)
    mock_form_ctx.__exit__ = MagicMock(return_value=False)

    def _make_col() -> MagicMock:
        col = MagicMock()
        col.__enter__ = MagicMock(return_value=col)
        col.__exit__ = MagicMock(return_value=False)
        return col

    def _cols_side_effect(spec) -> list:
        n = len(spec) if isinstance(spec, (list, tuple)) else int(spec)
        return [_make_col() for _ in range(n)]

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
        "expander": MagicMock(return_value=mock_expander_ctx),
        "text_input": MagicMock(return_value=""),
        "text_area": MagicMock(return_value=""),
        "selectbox": MagicMock(return_value="All"),
        "button": MagicMock(return_value=False),
        "form": MagicMock(return_value=mock_form_ctx),
        "form_submit_button": MagicMock(return_value=False),
        "rerun": MagicMock(),
    }
    for name, mock in st_mocks.items():
        monkeypatch.setattr(f"ui.pages.strategy_library.st.{name}", mock)

    monkeypatch.setattr("ui.pages.strategy_library.st.session_state", state)

    return {
        "state": state,
        "strategies": mock_strategies,
        "list_strategies": mock_list_strategies,
        "list_users": mock_list_users,
        "create_strategy": mock_create_strategy,
        "update_status": mock_update_status,
        "st": st_mocks,
    }


# ===========================================================================
# Tests — display
# ===========================================================================


class TestStrategyLibraryDisplay:
    def test_approved_strategy_rendered_as_expander(
        self, mock_sl: dict
    ) -> None:
        mock_sl["strategies"].append(_make_strategy(name="SMA Crossover", status="APPROVED"))
        strategy_library_page()
        mock_sl["st"]["expander"].assert_called_once()
        label = str(mock_sl["st"]["expander"].call_args[0][0])
        assert "SMA Crossover" in label

    def test_no_strategies_shows_info(self, mock_sl: dict) -> None:
        strategy_library_page()
        mock_sl["st"]["info"].assert_called_once()

    def test_expander_label_contains_strategy_status(
        self, mock_sl: dict
    ) -> None:
        mock_sl["strategies"].append(_make_strategy(status="APPROVED"))
        strategy_library_page()
        label = str(mock_sl["st"]["expander"].call_args[0][0])
        assert "APPROVED" in label

    def test_expander_label_contains_author_name(
        self, mock_sl: dict
    ) -> None:
        mock_sl["strategies"].append(_make_strategy(author_id=1))
        strategy_library_page()
        label = str(mock_sl["st"]["expander"].call_args[0][0])
        assert "Alice Analyst" in label

    def test_multiple_strategies_render_multiple_expanders(
        self, mock_sl: dict
    ) -> None:
        mock_sl["strategies"].extend([
            _make_strategy(1, "SMA"),
            _make_strategy(2, "RSI"),
        ])
        strategy_library_page()
        assert mock_sl["st"]["expander"].call_count == 2


# ===========================================================================
# Tests — filtering
# ===========================================================================


class TestStrategyLibraryFiltering:
    def test_search_by_name_narrows_results(self, mock_sl: dict) -> None:
        mock_sl["strategies"].extend([
            _make_strategy(1, "SMA Crossover", status="APPROVED"),
            _make_strategy(2, "RSI Momentum", status="APPROVED"),
        ])
        mock_sl["st"]["text_input"].return_value = "SMA"
        strategy_library_page()
        expander_labels = [
            str(c[0][0]) for c in mock_sl["st"]["expander"].call_args_list
        ]
        assert any("SMA Crossover" in l for l in expander_labels)
        assert not any("RSI Momentum" in l for l in expander_labels)

    def test_search_by_description_narrows_results(
        self, mock_sl: dict
    ) -> None:
        mock_sl["strategies"].extend([
            _make_strategy(
                1, "SMA", description="Uses moving averages", status="APPROVED"
            ),
            _make_strategy(
                2, "RSI", description="Uses relative strength index", status="APPROVED"
            ),
        ])
        mock_sl["st"]["text_input"].return_value = "moving"
        strategy_library_page()
        expander_labels = [
            str(c[0][0]) for c in mock_sl["st"]["expander"].call_args_list
        ]
        assert any("SMA" in l for l in expander_labels)
        assert not any("RSI" in l for l in expander_labels)

    def test_non_admin_sees_only_approved_strategies(
        self, mock_sl: dict
    ) -> None:
        mock_sl["strategies"].extend([
            _make_strategy(1, "SMA", status="APPROVED"),
            _make_strategy(2, "New Strategy", status="PENDING"),
        ])
        # User is analyst (default) — should see only APPROVED
        strategy_library_page()
        expander_labels = [
            str(c[0][0]) for c in mock_sl["st"]["expander"].call_args_list
        ]
        assert any("SMA" in l for l in expander_labels)
        assert not any("New Strategy" in l for l in expander_labels)

    def test_admin_status_filter_pending_shows_only_pending(
        self, mock_sl: dict, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "ui.pages.strategy_library.get_current_user",
            MagicMock(return_value=_ADMIN_USER),
        )
        mock_sl["strategies"].extend([
            _make_strategy(1, "SMA", status="APPROVED"),
            _make_strategy(2, "New Strategy", status="PENDING"),
        ])
        mock_sl["st"]["selectbox"].return_value = "PENDING"
        strategy_library_page()
        expander_labels = [
            str(c[0][0]) for c in mock_sl["st"]["expander"].call_args_list
        ]
        assert not any("SMA" in l for l in expander_labels)
        assert any("New Strategy" in l for l in expander_labels)

    def test_non_admin_does_not_see_status_selectbox(
        self, mock_sl: dict
    ) -> None:
        strategy_library_page()
        # Analyst should not trigger selectbox for status filter
        selectbox_calls = [
            c for c in mock_sl["st"]["selectbox"].call_args_list
            if c.kwargs.get("key") == "sl_status"
        ]
        assert len(selectbox_calls) == 0


# ===========================================================================
# Tests — run-this navigation
# ===========================================================================


class TestRunThisNavigation:
    def test_run_this_button_sets_page_to_new_run(
        self, mock_sl: dict
    ) -> None:
        strat = _make_strategy(strategy_id=5)
        mock_sl["strategies"].append(strat)

        def _btn(*args, key: str = "", **kwargs):
            return key == f"sl_run_{strat.id}"

        mock_sl["st"]["button"].side_effect = _btn
        strategy_library_page()
        assert mock_sl["state"]["page"] == "new_run"

    def test_run_this_calls_rerun(self, mock_sl: dict) -> None:
        strat = _make_strategy(strategy_id=7)
        mock_sl["strategies"].append(strat)

        def _btn(*args, key: str = "", **kwargs):
            return key == f"sl_run_{strat.id}"

        mock_sl["st"]["button"].side_effect = _btn
        strategy_library_page()
        mock_sl["st"]["rerun"].assert_called()


# ===========================================================================
# Tests — admin approval
# ===========================================================================


class TestAdminApproval:
    def test_approve_button_calls_update_strategy_status(
        self, mock_sl: dict, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "ui.pages.strategy_library.get_current_user",
            MagicMock(return_value=_ADMIN_USER),
        )
        pending = _make_strategy(strategy_id=10, name="Pending Strat", status="PENDING")
        mock_sl["strategies"].append(pending)
        mock_sl["st"]["selectbox"].return_value = "PENDING"

        def _btn(*args, key: str = "", **kwargs):
            return key == f"sl_approve_{pending.id}"

        mock_sl["st"]["button"].side_effect = _btn
        strategy_library_page()
        mock_sl["update_status"].assert_called_once()
        args = mock_sl["update_status"].call_args[0]
        # args: (db, strategy_id, new_status)
        assert args[1] == pending.id
        assert args[2] == "APPROVED"

    def test_approve_button_not_shown_for_non_admin(
        self, mock_sl: dict
    ) -> None:
        pending = _make_strategy(strategy_id=11, status="PENDING")
        mock_sl["strategies"].append(pending)
        strategy_library_page()
        # Analyst cannot approve — update_strategy_status should not be called
        # regardless of button mock state
        mock_sl["update_status"].assert_not_called()


# ===========================================================================
# Tests — submission form
# ===========================================================================


class TestStrategySubmissionForm:
    def test_form_shown_for_analyst(self, mock_sl: dict) -> None:
        strategy_library_page()
        mock_sl["st"]["form"].assert_called_once()

    def test_form_shown_for_admin(
        self, mock_sl: dict, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "ui.pages.strategy_library.get_current_user",
            MagicMock(return_value=_ADMIN_USER),
        )
        strategy_library_page()
        mock_sl["st"]["form"].assert_called_once()

    def test_form_hidden_for_viewer(
        self, mock_sl: dict, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "ui.pages.strategy_library.get_current_user",
            MagicMock(return_value=_VIEWER_USER),
        )
        strategy_library_page()
        mock_sl["st"]["form"].assert_not_called()

    def test_empty_name_shows_error(self, mock_sl: dict) -> None:
        mock_sl["st"]["form_submit_button"].return_value = True
        mock_sl["st"]["text_input"].return_value = ""   # empty name
        mock_sl["st"]["text_area"].return_value = "def run(): pass"
        strategy_library_page()
        mock_sl["st"]["error"].assert_called_once()
        mock_sl["create_strategy"].assert_not_called()

    def test_empty_code_shows_error(self, mock_sl: dict) -> None:
        mock_sl["st"]["form_submit_button"].return_value = True
        mock_sl["st"]["text_input"].return_value = "My Strategy"

        def _text_area(*args, key: str = "", **kwargs):
            return "" if key == "sl_code" else "A description"

        mock_sl["st"]["text_area"].side_effect = _text_area
        strategy_library_page()
        mock_sl["st"]["error"].assert_called_once()
        mock_sl["create_strategy"].assert_not_called()

    def test_valid_submission_calls_create_strategy(
        self, mock_sl: dict
    ) -> None:
        mock_sl["st"]["form_submit_button"].return_value = True
        mock_sl["st"]["text_input"].return_value = "My New Strategy"

        def _text_area(*args, key: str = "", **kwargs):
            return "def run(): pass" if key == "sl_code" else "Great description"

        mock_sl["st"]["text_area"].side_effect = _text_area
        strategy_library_page()
        mock_sl["create_strategy"].assert_called_once()

    def test_create_strategy_receives_correct_args(
        self, mock_sl: dict
    ) -> None:
        mock_sl["st"]["form_submit_button"].return_value = True
        mock_sl["st"]["text_input"].return_value = "  My Strategy  "

        def _text_area(*args, key: str = "", **kwargs):
            if key == "sl_code":
                return "  def run(): pass  "
            return "A description"

        mock_sl["st"]["text_area"].side_effect = _text_area
        strategy_library_page()
        call_kwargs = mock_sl["create_strategy"].call_args[1]
        args = mock_sl["create_strategy"].call_args[0]
        # args: (db, name, description, code, author_id) or kwargs
        name_arg = call_kwargs.get("name", args[1] if len(args) > 1 else None)
        code_arg = call_kwargs.get("code", args[3] if len(args) > 3 else None)
        # Name and code should be stripped
        assert name_arg == "My Strategy"
        assert code_arg == "def run(): pass"

    def test_valid_submission_calls_rerun(self, mock_sl: dict) -> None:
        mock_sl["st"]["form_submit_button"].return_value = True
        mock_sl["st"]["text_input"].return_value = "My Strategy"

        def _text_area(*args, key: str = "", **kwargs):
            return "def run(): pass" if key == "sl_code" else ""

        mock_sl["st"]["text_area"].side_effect = _text_area
        strategy_library_page()
        mock_sl["st"]["rerun"].assert_called()

    def test_no_submit_no_create_strategy(self, mock_sl: dict) -> None:
        # form_submit_button returns False (default)
        strategy_library_page()
        mock_sl["create_strategy"].assert_not_called()
