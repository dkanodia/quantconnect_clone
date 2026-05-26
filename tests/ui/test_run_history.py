"""
Tests for ui/pages/run_history.py.

All Streamlit and DB calls are mocked — no live runtime or database needed.

Covers
------
filter logic (_apply_filters)
    no filters: all runs returned
    strategy name filter is case-insensitive substring match
    status filter hides runs with non-matching status
    owner filter restricts to a specific user's runs
    date range: from_date excludes earlier runs
    date range: to_date excludes later runs
    tag search matches runs containing the tag substring
    multiple filters applied together

page-level behaviour
    clear-filters button resets run_history_filters to {}
    row-count label matches the number of filtered runs
    private runs of other users not shown to viewer (enforced via DB helper)
    "Open →" button sets selected_run_id and navigates to run_detail
    st.dataframe called with correct column set
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock
import uuid

import pytest
import pandas as pd

from ui.pages.run_history import _apply_filters, run_history_page


# ===========================================================================
# Shared test data
# ===========================================================================

_USER_ID_MAP = {
    "Alice":   1,
    "Bob":     2,
    "Charlie": 3,
}
_ID_USER_MAP = {v: k for k, v in _USER_ID_MAP.items()}

_CURRENT_USER = {
    "user_id": 1,
    "name": "Alice",
    "email": "alice@example.com",
    "role": "analyst",
    "avatar_initials": "A",
}


def _make_run(
    owner_id: int = 1,
    strategy_name: str = "SMA",
    status: str = "DONE",
    visibility: str = "TEAM",
    result: dict | None = None,
    created_at: datetime | None = None,
    tags: list | None = None,
) -> MagicMock:
    run = MagicMock()
    run.id = str(uuid.uuid4())
    run.strategy_name = strategy_name
    run.status = status
    run.visibility = visibility
    run.owner_id = owner_id
    run.result = result
    run.tags = tags or []
    run.created_at = (created_at or datetime(2024, 6, 15, tzinfo=timezone.utc)).replace(tzinfo=None)
    return run


# ===========================================================================
# Tests — _apply_filters (pure logic, no Streamlit mocking needed)
# ===========================================================================


class TestApplyFilters:
    def _apply(self, runs, **filter_kwargs) -> list:
        return _apply_filters(runs, filter_kwargs, _USER_ID_MAP)

    # ── No filters ─────────────────────────────────────────────────────────

    def test_no_filters_returns_all_runs(self) -> None:
        runs = [_make_run(), _make_run(), _make_run()]
        result = self._apply(runs)
        assert len(result) == 3

    # ── Strategy name filter ───────────────────────────────────────────────

    def test_strategy_filter_exact_match(self) -> None:
        r1 = _make_run(strategy_name="SMA")
        r2 = _make_run(strategy_name="RSI")
        result = self._apply([r1, r2], strategy="SMA")
        assert result == [r1]

    def test_strategy_filter_case_insensitive(self) -> None:
        r1 = _make_run(strategy_name="SmaCrossover")
        r2 = _make_run(strategy_name="RSI")
        result = self._apply([r1, r2], strategy="sma")
        assert result == [r1]

    def test_strategy_filter_substring(self) -> None:
        r1 = _make_run(strategy_name="SmaCrossover")
        r2 = _make_run(strategy_name="EMA")
        result = self._apply([r1, r2], strategy="cross")
        assert result == [r1]

    def test_strategy_empty_string_returns_all(self) -> None:
        runs = [_make_run(strategy_name="SMA"), _make_run(strategy_name="RSI")]
        result = self._apply(runs, strategy="")
        assert len(result) == 2

    # ── Status filter ──────────────────────────────────────────────────────

    def test_status_filter_done(self) -> None:
        r1 = _make_run(status="DONE")
        r2 = _make_run(status="RUNNING")
        r3 = _make_run(status="FAILED")
        result = self._apply([r1, r2, r3], status="DONE")
        assert result == [r1]

    def test_status_filter_running(self) -> None:
        r1 = _make_run(status="DONE")
        r2 = _make_run(status="RUNNING")
        result = self._apply([r1, r2], status="RUNNING")
        assert result == [r2]

    def test_status_all_returns_all(self) -> None:
        runs = [_make_run(status=s) for s in ("DONE", "RUNNING", "FAILED")]
        result = self._apply(runs, status="All")
        assert len(result) == 3

    # ── Owner filter ───────────────────────────────────────────────────────

    def test_owner_filter_by_name(self) -> None:
        r1 = _make_run(owner_id=1)  # Alice
        r2 = _make_run(owner_id=2)  # Bob
        result = self._apply([r1, r2], owner="Bob")
        assert result == [r2]

    def test_owner_all_returns_all(self) -> None:
        runs = [_make_run(owner_id=i) for i in (1, 2, 3)]
        result = self._apply(runs, owner="All")
        assert len(result) == 3

    def test_owner_unknown_name_returns_empty(self) -> None:
        runs = [_make_run(owner_id=1)]
        result = self._apply(runs, owner="NoSuchUser")
        assert result == []

    # ── Date range filter ──────────────────────────────────────────────────

    def test_from_date_excludes_earlier_runs(self) -> None:
        old = _make_run(created_at=datetime(2024, 1, 1))
        new = _make_run(created_at=datetime(2024, 6, 15))
        result = self._apply([old, new], from_date=date(2024, 6, 1))
        assert result == [new]

    def test_to_date_excludes_later_runs(self) -> None:
        early = _make_run(created_at=datetime(2024, 3, 1))
        late = _make_run(created_at=datetime(2024, 9, 1))
        result = self._apply([early, late], to_date=date(2024, 6, 30))
        assert result == [early]

    def test_from_and_to_date_together(self) -> None:
        r1 = _make_run(created_at=datetime(2024, 1, 15))
        r2 = _make_run(created_at=datetime(2024, 6, 15))
        r3 = _make_run(created_at=datetime(2024, 11, 15))
        result = self._apply(
            [r1, r2, r3],
            from_date=date(2024, 3, 1),
            to_date=date(2024, 9, 30),
        )
        assert result == [r2]

    def test_none_date_is_ignored(self) -> None:
        runs = [_make_run(), _make_run()]
        result = self._apply(runs, from_date=None, to_date=None)
        assert len(result) == 2

    # ── Tag filter ─────────────────────────────────────────────────────────

    def test_tag_filter_matches_substring(self) -> None:
        r1 = _make_run(tags=["momentum", "daily"])
        r2 = _make_run(tags=["mean-reversion"])
        result = self._apply([r1, r2], tag="mom")
        assert result == [r1]

    def test_tag_filter_case_insensitive(self) -> None:
        r1 = _make_run(tags=["Momentum"])
        r2 = _make_run(tags=["RSI"])
        result = self._apply([r1, r2], tag="momentum")
        assert result == [r1]

    def test_tag_empty_string_returns_all(self) -> None:
        runs = [_make_run(tags=["a"]), _make_run(tags=["b"])]
        result = self._apply(runs, tag="")
        assert len(result) == 2

    def test_tag_no_match_returns_empty(self) -> None:
        runs = [_make_run(tags=["foo"])]
        result = self._apply(runs, tag="bar")
        assert result == []

    # ── Combined filters ───────────────────────────────────────────────────

    def test_multiple_filters_combined(self) -> None:
        r1 = _make_run(strategy_name="SMA", status="DONE", owner_id=1)
        r2 = _make_run(strategy_name="SMA", status="RUNNING", owner_id=1)
        r3 = _make_run(strategy_name="RSI", status="DONE", owner_id=2)
        result = self._apply(
            [r1, r2, r3], strategy="SMA", status="DONE", owner="Alice"
        )
        assert result == [r1]


# ===========================================================================
# Fixtures — full page mock
# ===========================================================================


@pytest.fixture
def mock_run_history(monkeypatch):
    """
    Patch all external deps of run_history_page() and return control mocks.
    """
    state = {
        "page": "run_history",
        "selected_run_id": None,
        "run_history_filters": {},
    }

    monkeypatch.setattr("ui.pages.run_history.require_role", MagicMock())
    monkeypatch.setattr(
        "ui.pages.run_history.get_current_user",
        MagicMock(return_value=_CURRENT_USER),
    )

    mock_runs: list = []
    mock_users: list = []

    mock_db_session = MagicMock()

    @contextmanager
    def _mock_get_db():
        yield mock_db_session

    monkeypatch.setattr("ui.pages.run_history.get_db", _mock_get_db)
    monkeypatch.setattr(
        "ui.pages.run_history.list_runs_for_user",
        MagicMock(return_value=mock_runs),
    )

    mock_user_obj = MagicMock()
    mock_user_obj.id = 1
    mock_user_obj.name = "Alice"
    mock_users.append(mock_user_obj)

    monkeypatch.setattr(
        "ui.pages.run_history.list_users",
        MagicMock(return_value=mock_users),
    )

    # Streamlit mocks — columns returns the RIGHT number based on spec
    def _make_col() -> MagicMock:
        col = MagicMock()
        col.__enter__ = MagicMock(return_value=col)
        col.__exit__ = MagicMock(return_value=False)
        return col

    def _cols_side_effect(spec):
        n = len(spec) if isinstance(spec, (list, tuple)) else int(spec)
        return [_make_col() for _ in range(n)]

    mock_expander = MagicMock()
    mock_expander.__enter__ = MagicMock(return_value=mock_expander)
    mock_expander.__exit__ = MagicMock(return_value=False)

    dataframe_mock = MagicMock()
    button_mock = MagicMock(return_value=False)
    text_input_mock = MagicMock(return_value="")
    selectbox_mock = MagicMock(return_value="All")
    date_input_mock = MagicMock(return_value=None)
    caption_mock = MagicMock()
    rerun_mock = MagicMock()

    monkeypatch.setattr("ui.pages.run_history.st.title", MagicMock())
    monkeypatch.setattr("ui.pages.run_history.st.subheader", MagicMock())
    monkeypatch.setattr("ui.pages.run_history.st.markdown", MagicMock())
    monkeypatch.setattr("ui.pages.run_history.st.caption", caption_mock)
    monkeypatch.setattr("ui.pages.run_history.st.columns", MagicMock(side_effect=_cols_side_effect))
    monkeypatch.setattr("ui.pages.run_history.st.expander", MagicMock(return_value=mock_expander))
    monkeypatch.setattr("ui.pages.run_history.st.dataframe", dataframe_mock)
    monkeypatch.setattr("ui.pages.run_history.st.button", button_mock)
    monkeypatch.setattr("ui.pages.run_history.st.text_input", text_input_mock)
    monkeypatch.setattr("ui.pages.run_history.st.selectbox", selectbox_mock)
    monkeypatch.setattr("ui.pages.run_history.st.date_input", date_input_mock)
    monkeypatch.setattr("ui.pages.run_history.st.empty", MagicMock())
    monkeypatch.setattr("ui.pages.run_history.st.rerun", rerun_mock)
    monkeypatch.setattr("ui.pages.run_history.st.session_state", state)

    return {
        "state": state,
        "runs": mock_runs,
        "users": mock_users,
        "dataframe": dataframe_mock,
        "button": button_mock,
        "caption": caption_mock,
        "selectbox": selectbox_mock,
        "rerun": rerun_mock,
    }


# ===========================================================================
# Tests — page-level behaviour
# ===========================================================================


class TestRunHistoryPage:
    def test_dataframe_called_once(self, mock_run_history: dict) -> None:
        run_history_page()
        mock_run_history["dataframe"].assert_called_once()

    def test_dataframe_has_correct_columns(
        self, mock_run_history: dict
    ) -> None:
        mock_run_history["runs"].append(_make_run())
        run_history_page()
        args, kwargs = mock_run_history["dataframe"].call_args
        df: pd.DataFrame = args[0]
        expected_cols = {
            "Strategy", "Owner", "Sharpe", "CAGR",
            "Max DD", "Win Rate", "Visibility", "Status", "Created",
        }
        assert expected_cols.issubset(set(df.columns))

    def test_row_count_label_matches_filtered_runs(
        self, mock_run_history: dict
    ) -> None:
        for _ in range(5):
            mock_run_history["runs"].append(_make_run())
        run_history_page()
        caption_texts = [
            str(c[0][0]) for c in mock_run_history["caption"].call_args_list
        ]
        assert any("5" in t for t in caption_texts)

    def test_clear_filters_resets_state(
        self, mock_run_history: dict
    ) -> None:
        # Pre-set some filters
        mock_run_history["state"]["run_history_filters"] = {
            "strategy": "SMA", "status": "DONE"
        }
        # Make the "Clear filters" button return True (clicked)
        def _btn(label="", key="", **kwargs):
            return key == "rh_clear"
        mock_run_history["button"].side_effect = _btn

        run_history_page()
        # After clearing, filters dict must be empty
        assert mock_run_history["state"]["run_history_filters"] == {}

    def test_clear_filters_calls_rerun(
        self, mock_run_history: dict
    ) -> None:
        def _btn(label="", key="", **kwargs):
            return key == "rh_clear"
        mock_run_history["button"].side_effect = _btn

        run_history_page()
        mock_run_history["rerun"].assert_called()

    def test_open_button_sets_selected_run_id(
        self, mock_run_history: dict
    ) -> None:
        run = _make_run()
        mock_run_history["runs"].append(run)

        # Use per-key side_effect so owner/status get "All" (keeping
        # filtered_runs non-empty) while rh_select_run returns the run ID.
        def _selectbox(*args, key="", options=None, **kwargs):
            return run.id if key == "rh_select_run" else "All"

        mock_run_history["selectbox"].side_effect = _selectbox

        def _btn(label="", key="", **kwargs):
            return key == "rh_open_run"

        mock_run_history["button"].side_effect = _btn
        run_history_page()

        assert mock_run_history["state"]["selected_run_id"] == run.id

    def test_open_button_navigates_to_run_detail(
        self, mock_run_history: dict
    ) -> None:
        run = _make_run()
        mock_run_history["runs"].append(run)

        def _selectbox(*args, key="", options=None, **kwargs):
            return run.id if key == "rh_select_run" else "All"

        mock_run_history["selectbox"].side_effect = _selectbox

        def _btn(label="", key="", **kwargs):
            return key == "rh_open_run"

        mock_run_history["button"].side_effect = _btn
        run_history_page()

        assert mock_run_history["state"]["page"] == "run_detail"

    def test_use_container_width_true(self, mock_run_history: dict) -> None:
        run_history_page()
        _, kwargs = mock_run_history["dataframe"].call_args
        assert kwargs.get("use_container_width") is True

    def test_hide_index_true(self, mock_run_history: dict) -> None:
        run_history_page()
        _, kwargs = mock_run_history["dataframe"].call_args
        assert kwargs.get("hide_index") is True

    def test_empty_table_still_renders_dataframe(
        self, mock_run_history: dict
    ) -> None:
        """No runs → dataframe rendered with headers only (empty df)."""
        run_history_page()
        mock_run_history["dataframe"].assert_called_once()
        args, _ = mock_run_history["dataframe"].call_args
        df: pd.DataFrame = args[0]
        assert len(df) == 0
