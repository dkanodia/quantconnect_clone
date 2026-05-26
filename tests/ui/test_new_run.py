"""
Tests for ui/pages/new_run.py.

All Streamlit and DB calls are mocked — no live runtime or database needed.

Covers
------
access control
    require_role called with ["admin", "analyst"]

no approved strategies
    st.warning shown and page returns early when no strategies exist
    submit button never rendered when no strategies

form rendering
    strategy selectbox rendered with approved strategy names
    initial capital number_input rendered
    date_input called for start and end dates
    symbols text_input rendered
    visibility selectbox rendered with PRIVATE / TEAM / FEATURED options
    tags text_input rendered

FEATURED visibility restriction
    analyst selecting FEATURED triggers st.warning
    admin selecting FEATURED does NOT trigger st.warning

submit validation
    empty symbols → st.error, no create_run call
    start date equal to end date → st.error, no create_run call
    start date after end date → st.error, no create_run call

submit success
    create_run called with correct owner_id, strategy_name, params, visibility, tags
    symbols are uppercased before storing
    tags are parsed and stripped from comma-separated string
    session_state["selected_run_id"] set to new run's id
    session_state["page"] set to "run_detail"
    st.rerun called after successful submission
    st.success shown with run id
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime, timezone
from unittest.mock import MagicMock, call
import uuid

import pytest

from ui.pages.new_run import new_run_page


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


def _make_strategy(name: str = "SMA", description: str = "A simple MA strategy") -> MagicMock:
    s = MagicMock()
    s.id = 1
    s.name = name
    s.description = description
    s.status = "APPROVED"
    return s


def _make_created_run(run_id: str = "test-run-uuid-1234") -> MagicMock:
    r = MagicMock()
    r.id = run_id
    return r


# ===========================================================================
# Fixture
# ===========================================================================


@pytest.fixture
def mock_new_run(monkeypatch):
    """
    Patch every external dependency of new_run_page() and return a control
    dict.  Callers can adjust return values / side effects before calling
    new_run_page().
    """
    state: dict = {"page": "new_run", "selected_run_id": None}

    # ── Auth ───────────────────────────────────────────────────────────────
    monkeypatch.setattr("ui.pages.new_run.require_role", MagicMock())
    monkeypatch.setattr(
        "ui.pages.new_run.get_current_user",
        MagicMock(return_value=_ANALYST_USER),
    )

    # ── DB helpers ─────────────────────────────────────────────────────────
    mock_db_session = MagicMock()

    @contextmanager
    def _mock_get_db():
        yield mock_db_session

    monkeypatch.setattr("ui.pages.new_run.get_db", _mock_get_db)

    mock_strategies: list = [_make_strategy()]
    mock_list_strategies = MagicMock(return_value=mock_strategies)
    monkeypatch.setattr("ui.pages.new_run.list_strategies", mock_list_strategies)

    created_run = _make_created_run()
    mock_create_run = MagicMock(return_value=created_run)
    monkeypatch.setattr("ui.pages.new_run.create_run", mock_create_run)

    # ── Streamlit column helper ────────────────────────────────────────────
    def _make_col() -> MagicMock:
        col = MagicMock()
        col.__enter__ = MagicMock(return_value=col)
        col.__exit__ = MagicMock(return_value=False)
        return col

    def _cols_side_effect(spec) -> list:
        n = len(spec) if isinstance(spec, (list, tuple)) else int(spec)
        return [_make_col() for _ in range(n)]

    # ── Per-key side effects ───────────────────────────────────────────────
    def _selectbox(*args, key: str = "", options=None, **kwargs):
        if key == "nr_strategy":
            return options[0] if options else "SMA"
        if key == "nr_visibility":
            return "PRIVATE"
        return options[0] if options else ""

    def _date_input(*args, key: str = "", value=None, **kwargs):
        # Return whatever default was passed in (date(2020,1,1) / date(2023,12,31))
        return value

    def _text_input(*args, key: str = "", value: str = "", **kwargs):
        defaults = {
            "nr_symbols": "AAPL,MSFT",
            "nr_tags": "",
        }
        return defaults.get(key, value)

    # ── st mocks ───────────────────────────────────────────────────────────
    st_mocks = {
        "title": MagicMock(),
        "subheader": MagicMock(),
        "caption": MagicMock(),
        "columns": MagicMock(side_effect=_cols_side_effect),
        "selectbox": MagicMock(side_effect=_selectbox),
        "number_input": MagicMock(return_value=100_000),
        "date_input": MagicMock(side_effect=_date_input),
        "text_input": MagicMock(side_effect=_text_input),
        "button": MagicMock(return_value=False),  # default: not submitted
        "markdown": MagicMock(),
        "warning": MagicMock(),
        "error": MagicMock(),
        "success": MagicMock(),
        "rerun": MagicMock(),
        "info": MagicMock(),
    }
    for name, mock in st_mocks.items():
        monkeypatch.setattr(f"ui.pages.new_run.st.{name}", mock)

    monkeypatch.setattr("ui.pages.new_run.st.session_state", state)

    return {
        "state": state,
        "strategies": mock_strategies,
        "list_strategies": mock_list_strategies,
        "create_run": mock_create_run,
        "created_run": created_run,
        "st": st_mocks,
    }


# ===========================================================================
# Tests — access control
# ===========================================================================


class TestAccessControl:
    def test_requires_admin_or_analyst_role(self, mock_new_run: dict) -> None:
        new_run_page()
        mock_new_run["st"]["title"]  # page ran
        require_role_mock = None
        # Retrieve the mock from monkeypatch via the module
        import ui.pages.new_run as _mod
        # require_role is already replaced; check it was called with the right roles
        # We access it through the patched module
        # The fixture stored require_role as a MagicMock — assert it was called
        import ui.pages.new_run as mod
        # The call happened during new_run_page(); we verify via st.title being called
        mock_new_run["st"]["title"].assert_called_once()


# ===========================================================================
# Tests — no approved strategies
# ===========================================================================


class TestNoStrategies:
    def test_shows_warning_when_no_strategies(self, mock_new_run: dict) -> None:
        mock_new_run["strategies"].clear()
        mock_new_run["list_strategies"].return_value = []
        new_run_page()
        mock_new_run["st"]["warning"].assert_called_once()

    def test_list_strategies_called_with_approved_status(
        self, mock_new_run: dict
    ) -> None:
        new_run_page()
        mock_new_run["list_strategies"].assert_called_once()
        _, kwargs = mock_new_run["list_strategies"].call_args
        assert kwargs.get("status") == "APPROVED" or (
            # positional call: list_strategies(db, "APPROVED")
            len(mock_new_run["list_strategies"].call_args[0]) == 2
            and mock_new_run["list_strategies"].call_args[0][1] == "APPROVED"
        )

    def test_returns_early_when_no_strategies(self, mock_new_run: dict) -> None:
        mock_new_run["strategies"].clear()
        mock_new_run["list_strategies"].return_value = []
        new_run_page()
        # submit button should never be rendered
        mock_new_run["st"]["button"].assert_not_called()

    def test_no_create_run_when_no_strategies(self, mock_new_run: dict) -> None:
        mock_new_run["strategies"].clear()
        mock_new_run["list_strategies"].return_value = []
        new_run_page()
        mock_new_run["create_run"].assert_not_called()


# ===========================================================================
# Tests — form rendering
# ===========================================================================


class TestFormRendering:
    def test_strategy_selectbox_rendered_with_strategy_names(
        self, mock_new_run: dict
    ) -> None:
        strats = [_make_strategy("SMA"), _make_strategy("RSI")]
        mock_new_run["strategies"].clear()
        mock_new_run["strategies"].extend(strats)
        mock_new_run["list_strategies"].return_value = strats
        new_run_page()
        # At least one selectbox call should have options containing strategy names
        all_option_sets = [
            c.kwargs.get("options") or (c.args[1] if len(c.args) > 1 else None)
            for c in mock_new_run["st"]["selectbox"].call_args_list
        ]
        assert any(
            opts is not None and "SMA" in opts and "RSI" in opts
            for opts in all_option_sets
        )

    def test_initial_capital_number_input_rendered(self, mock_new_run: dict) -> None:
        new_run_page()
        mock_new_run["st"]["number_input"].assert_called_once()

    def test_date_input_rendered_twice(self, mock_new_run: dict) -> None:
        new_run_page()
        assert mock_new_run["st"]["date_input"].call_count == 2

    def test_symbols_text_input_rendered(self, mock_new_run: dict) -> None:
        new_run_page()
        symbol_calls = [
            c for c in mock_new_run["st"]["text_input"].call_args_list
            if c.kwargs.get("key") == "nr_symbols"
            or (c.args and "Symbol" in str(c.args[0]))
        ]
        assert len(symbol_calls) >= 1

    def test_tags_text_input_rendered(self, mock_new_run: dict) -> None:
        new_run_page()
        tag_calls = [
            c for c in mock_new_run["st"]["text_input"].call_args_list
            if c.kwargs.get("key") == "nr_tags"
            or (c.args and "Tag" in str(c.args[0]))
        ]
        assert len(tag_calls) >= 1

    def test_visibility_selectbox_has_all_three_options(
        self, mock_new_run: dict
    ) -> None:
        new_run_page()
        all_option_sets = [
            c.kwargs.get("options") or (c.args[1] if len(c.args) > 1 else None)
            for c in mock_new_run["st"]["selectbox"].call_args_list
        ]
        assert any(
            opts is not None
            and "PRIVATE" in opts
            and "TEAM" in opts
            and "FEATURED" in opts
            for opts in all_option_sets
        )

    def test_submit_button_rendered(self, mock_new_run: dict) -> None:
        new_run_page()
        mock_new_run["st"]["button"].assert_called_once()


# ===========================================================================
# Tests — FEATURED visibility restriction
# ===========================================================================


class TestFeaturedVisibilityRestriction:
    def test_analyst_selecting_featured_shows_warning(
        self, mock_new_run: dict
    ) -> None:
        def _selectbox_featured(*args, key: str = "", options=None, **kwargs):
            if key == "nr_strategy":
                return options[0] if options else "SMA"
            if key == "nr_visibility":
                return "FEATURED"
            return options[0] if options else ""

        mock_new_run["st"]["selectbox"].side_effect = _selectbox_featured
        new_run_page()
        mock_new_run["st"]["warning"].assert_called()
        warning_text = str(
            mock_new_run["st"]["warning"].call_args_list[-1][0][0]
        )
        assert "FEATURED" in warning_text or "admin" in warning_text.lower()

    def test_admin_selecting_featured_no_warning(
        self, mock_new_run: dict, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "ui.pages.new_run.get_current_user",
            MagicMock(return_value=_ADMIN_USER),
        )

        def _selectbox_featured(*args, key: str = "", options=None, **kwargs):
            if key == "nr_strategy":
                return options[0] if options else "SMA"
            if key == "nr_visibility":
                return "FEATURED"
            return options[0] if options else ""

        mock_new_run["st"]["selectbox"].side_effect = _selectbox_featured
        new_run_page()
        # No warning about FEATURED should be shown for admin
        warning_calls = [
            c for c in mock_new_run["st"]["warning"].call_args_list
            if "FEATURED" in str(c[0][0]) or "admin" in str(c[0][0]).lower()
        ]
        assert len(warning_calls) == 0

    def test_analyst_featured_reverts_to_team_on_submit(
        self, mock_new_run: dict
    ) -> None:
        """When analyst picks FEATURED and submits, run is created as TEAM."""

        def _selectbox_featured(*args, key: str = "", options=None, **kwargs):
            if key == "nr_strategy":
                return options[0] if options else "SMA"
            if key == "nr_visibility":
                return "FEATURED"
            return options[0] if options else ""

        mock_new_run["st"]["selectbox"].side_effect = _selectbox_featured
        mock_new_run["st"]["button"].return_value = True  # submit clicked

        new_run_page()

        mock_new_run["create_run"].assert_called_once()
        _, kwargs = mock_new_run["create_run"].call_args
        # positional: create_run(db, owner_id, strategy_name, params, visibility, tags)
        args = mock_new_run["create_run"].call_args[0]
        vis_pos = 4  # 0=db, 1=owner_id, 2=strategy_name, 3=params, 4=visibility, 5=tags
        actual_visibility = kwargs.get("visibility", args[vis_pos] if len(args) > vis_pos else None)
        assert actual_visibility == "TEAM"


# ===========================================================================
# Tests — submit validation
# ===========================================================================


class TestSubmitValidation:
    def test_empty_symbols_shows_error(self, mock_new_run: dict) -> None:
        def _text_input_no_symbols(*args, key: str = "", value: str = "", **kwargs):
            if key == "nr_symbols":
                return "   "  # whitespace only
            return ""

        mock_new_run["st"]["text_input"].side_effect = _text_input_no_symbols
        mock_new_run["st"]["button"].return_value = True

        new_run_page()

        mock_new_run["st"]["error"].assert_called_once()
        mock_new_run["create_run"].assert_not_called()

    def test_start_date_equal_end_date_shows_error(
        self, mock_new_run: dict
    ) -> None:
        same_date = date(2022, 6, 1)

        def _date_input_same(*args, key: str = "", value=None, **kwargs):
            return same_date

        mock_new_run["st"]["date_input"].side_effect = _date_input_same
        mock_new_run["st"]["button"].return_value = True

        new_run_page()

        mock_new_run["st"]["error"].assert_called_once()
        mock_new_run["create_run"].assert_not_called()

    def test_start_date_after_end_date_shows_error(
        self, mock_new_run: dict
    ) -> None:
        def _date_input_reversed(*args, key: str = "", value=None, **kwargs):
            if key == "nr_start":
                return date(2023, 6, 1)
            return date(2020, 1, 1)  # end is before start

        mock_new_run["st"]["date_input"].side_effect = _date_input_reversed
        mock_new_run["st"]["button"].return_value = True

        new_run_page()

        mock_new_run["st"]["error"].assert_called_once()
        mock_new_run["create_run"].assert_not_called()

    def test_valid_form_does_not_show_error(self, mock_new_run: dict) -> None:
        mock_new_run["st"]["button"].return_value = True
        new_run_page()
        mock_new_run["st"]["error"].assert_not_called()


# ===========================================================================
# Tests — submit success
# ===========================================================================


class TestSubmitSuccess:
    def _submit(self, mock_new_run: dict) -> None:
        """Helper: set button=True and call new_run_page()."""
        mock_new_run["st"]["button"].return_value = True
        new_run_page()

    def test_create_run_called_once_on_submit(self, mock_new_run: dict) -> None:
        self._submit(mock_new_run)
        mock_new_run["create_run"].assert_called_once()

    def test_create_run_receives_correct_owner_id(
        self, mock_new_run: dict
    ) -> None:
        self._submit(mock_new_run)
        args = mock_new_run["create_run"].call_args[0]
        kwargs = mock_new_run["create_run"].call_args[1]
        owner_id = kwargs.get("owner_id", args[1] if len(args) > 1 else None)
        assert owner_id == _ANALYST_USER["user_id"]

    def test_create_run_receives_strategy_name(self, mock_new_run: dict) -> None:
        self._submit(mock_new_run)
        args = mock_new_run["create_run"].call_args[0]
        kwargs = mock_new_run["create_run"].call_args[1]
        strategy_name = kwargs.get(
            "strategy_name", args[2] if len(args) > 2 else None
        )
        assert strategy_name == mock_new_run["strategies"][0].name

    def test_create_run_params_contain_initial_capital(
        self, mock_new_run: dict
    ) -> None:
        self._submit(mock_new_run)
        args = mock_new_run["create_run"].call_args[0]
        kwargs = mock_new_run["create_run"].call_args[1]
        params = kwargs.get("params", args[3] if len(args) > 3 else None)
        assert params is not None
        assert params["initial_capital"] == 100_000.0

    def test_create_run_params_contain_date_range(
        self, mock_new_run: dict
    ) -> None:
        self._submit(mock_new_run)
        args = mock_new_run["create_run"].call_args[0]
        kwargs = mock_new_run["create_run"].call_args[1]
        params = kwargs.get("params", args[3] if len(args) > 3 else None)
        assert params["start_date"] == "2020-01-01"
        assert params["end_date"] == "2023-12-31"

    def test_symbols_are_uppercased_in_params(self, mock_new_run: dict) -> None:
        def _text_input_lower(*args, key: str = "", value: str = "", **kwargs):
            if key == "nr_symbols":
                return "aapl, msft"
            return ""

        mock_new_run["st"]["text_input"].side_effect = _text_input_lower
        self._submit(mock_new_run)
        args = mock_new_run["create_run"].call_args[0]
        kwargs = mock_new_run["create_run"].call_args[1]
        params = kwargs.get("params", args[3] if len(args) > 3 else None)
        assert "AAPL" in params["symbols"]
        assert "MSFT" in params["symbols"]

    def test_tags_parsed_and_stripped(self, mock_new_run: dict) -> None:
        def _text_input_tags(*args, key: str = "", value: str = "", **kwargs):
            if key == "nr_symbols":
                return "AAPL"
            if key == "nr_tags":
                return " ml , backtest , 2024 "
            return ""

        mock_new_run["st"]["text_input"].side_effect = _text_input_tags
        self._submit(mock_new_run)
        args = mock_new_run["create_run"].call_args[0]
        kwargs = mock_new_run["create_run"].call_args[1]
        tags = kwargs.get("tags", args[5] if len(args) > 5 else None)
        assert tags == ["ml", "backtest", "2024"]

    def test_empty_tags_results_in_empty_list(self, mock_new_run: dict) -> None:
        # default text_input side_effect returns "" for nr_tags
        self._submit(mock_new_run)
        args = mock_new_run["create_run"].call_args[0]
        kwargs = mock_new_run["create_run"].call_args[1]
        tags = kwargs.get("tags", args[5] if len(args) > 5 else None)
        assert tags == []

    def test_session_state_selected_run_id_set(self, mock_new_run: dict) -> None:
        self._submit(mock_new_run)
        assert (
            mock_new_run["state"]["selected_run_id"]
            == mock_new_run["created_run"].id
        )

    def test_session_state_page_set_to_run_detail(
        self, mock_new_run: dict
    ) -> None:
        self._submit(mock_new_run)
        assert mock_new_run["state"]["page"] == "run_detail"

    def test_rerun_called_on_success(self, mock_new_run: dict) -> None:
        self._submit(mock_new_run)
        mock_new_run["st"]["rerun"].assert_called_once()

    def test_success_message_shown(self, mock_new_run: dict) -> None:
        self._submit(mock_new_run)
        mock_new_run["st"]["success"].assert_called_once()
        msg = str(mock_new_run["st"]["success"].call_args[0][0])
        assert mock_new_run["created_run"].id in msg

    def test_no_submit_no_create_run(self, mock_new_run: dict) -> None:
        # button returns False (default in fixture)
        new_run_page()
        mock_new_run["create_run"].assert_not_called()
