"""
Tests for ui/pages/compare.py.

All Streamlit and DB calls are mocked — no live runtime or database needed.

Covers
------
run selection
    multiselect rendered with ids of all visible runs
    preselected ids from session_state["compare_run_ids"] passed as default
    fewer than 2 selections shows info and returns early
    more than 4 selections shows warning and truncates to 4

metric table
    st.dataframe rendered when 2 or more runs selected
    dataframe includes expected column names (Strategy, Sharpe, CAGR, etc.)
    dash shown for metrics when run has no result

equity curves
    equity_chart called when at least one selected run has equity_curve data
    first selected run's equity_curve is passed as the primary argument
    additional runs with equity data appear in compare_curves
    equity_chart NOT called when no selected run has equity_curve data
    caption shown when no equity data available
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from unittest.mock import MagicMock
import uuid

import pytest

from ui.pages.compare import compare_page


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

_EQUITY_CURVE_A = {"2020-01-01": 100_000, "2020-06-01": 110_000, "2020-12-31": 120_000}
_EQUITY_CURVE_B = {"2020-01-01": 100_000, "2020-06-01": 105_000, "2020-12-31": 108_000}

_METRICS = {
    "sharpe_ratio": 1.5,
    "cagr": 0.20,
    "max_drawdown": -0.10,
    "win_rate": 0.55,
}


def _make_run(
    run_id: str | None = None,
    strategy_name: str = "SMA",
    status: str = "DONE",
    result: dict | None = None,
) -> MagicMock:
    r = MagicMock()
    r.id = run_id or str(uuid.uuid4())
    r.strategy_name = strategy_name
    r.status = status
    r.result = result
    r.owner_id = 1
    r.visibility = "TEAM"
    r.tags = []
    r.created_at = datetime.now(timezone.utc).replace(tzinfo=None)
    return r


# ===========================================================================
# Fixture
# ===========================================================================


@pytest.fixture
def mock_compare(monkeypatch):
    """
    Patch every external dependency of compare_page() and return a control
    dict.  Callers configure multiselect return and visible run list.
    """
    state: dict = {"page": "compare", "compare_run_ids": []}

    # ── Auth ───────────────────────────────────────────────────────────────
    monkeypatch.setattr("ui.pages.compare.require_role", MagicMock())
    monkeypatch.setattr(
        "ui.pages.compare.get_current_user",
        MagicMock(return_value=_ANALYST_USER),
    )

    # ── DB ─────────────────────────────────────────────────────────────────
    mock_db = MagicMock()

    @contextmanager
    def _mock_get_db():
        yield mock_db

    monkeypatch.setattr("ui.pages.compare.get_db", _mock_get_db)

    mock_visible: list = []
    mock_list_runs = MagicMock(return_value=mock_visible)
    monkeypatch.setattr("ui.pages.compare.list_runs_for_user", mock_list_runs)

    # ── equity_chart ───────────────────────────────────────────────────────
    mock_equity_chart = MagicMock()
    monkeypatch.setattr("ui.pages.compare.equity_chart", mock_equity_chart)

    # ── Streamlit mocks ────────────────────────────────────────────────────
    st_mocks = {
        "title": MagicMock(),
        "subheader": MagicMock(),
        "caption": MagicMock(),
        "info": MagicMock(),
        "warning": MagicMock(),
        "dataframe": MagicMock(),
        "multiselect": MagicMock(return_value=[]),  # default: nothing selected
        "markdown": MagicMock(),
        "error": MagicMock(),
        "success": MagicMock(),
        "rerun": MagicMock(),
    }
    for name, mock in st_mocks.items():
        monkeypatch.setattr(f"ui.pages.compare.st.{name}", mock)

    monkeypatch.setattr("ui.pages.compare.st.session_state", state)

    return {
        "state": state,
        "visible": mock_visible,
        "list_runs": mock_list_runs,
        "equity_chart": mock_equity_chart,
        "st": st_mocks,
    }


def _set_selected(mock_compare: dict, runs: list) -> None:
    """Helper: add runs to visible list and configure multiselect to select them."""
    mock_compare["visible"].extend(runs)
    mock_compare["st"]["multiselect"].return_value = [r.id for r in runs]


# ===========================================================================
# Tests — run selection
# ===========================================================================


class TestCompareRunSelection:
    def test_multiselect_rendered_with_all_visible_run_ids(
        self, mock_compare: dict
    ) -> None:
        run1 = _make_run("id-1", "SMA")
        run2 = _make_run("id-2", "RSI")
        mock_compare["visible"].extend([run1, run2])
        compare_page()
        call = mock_compare["st"]["multiselect"].call_args
        options = call.kwargs.get("options") or call.args[1]
        assert "id-1" in options
        assert "id-2" in options

    def test_preselected_ids_from_session_state_passed_as_default(
        self, mock_compare: dict
    ) -> None:
        run1 = _make_run("id-1")
        mock_compare["visible"].append(run1)
        mock_compare["state"]["compare_run_ids"] = ["id-1"]
        compare_page()
        call = mock_compare["st"]["multiselect"].call_args
        default = call.kwargs.get("default") or (call.args[2] if len(call.args) > 2 else [])
        assert "id-1" in default

    def test_zero_selections_shows_info(self, mock_compare: dict) -> None:
        mock_compare["st"]["multiselect"].return_value = []
        compare_page()
        mock_compare["st"]["info"].assert_called_once()

    def test_one_selection_shows_info_not_table(
        self, mock_compare: dict
    ) -> None:
        run = _make_run("id-1")
        mock_compare["visible"].append(run)
        mock_compare["st"]["multiselect"].return_value = ["id-1"]
        compare_page()
        mock_compare["st"]["info"].assert_called_once()
        mock_compare["st"]["dataframe"].assert_not_called()

    def test_more_than_four_shows_warning(self, mock_compare: dict) -> None:
        runs = [_make_run(f"id-{i}") for i in range(5)]
        _set_selected(mock_compare, runs)
        compare_page()
        mock_compare["st"]["warning"].assert_called_once()

    def test_more_than_four_truncates_to_first_four(
        self, mock_compare: dict
    ) -> None:
        runs = [
            _make_run(f"id-{i}", result={"metrics": _METRICS})
            for i in range(5)
        ]
        _set_selected(mock_compare, runs)
        compare_page()
        # dataframe should be called (some runs were selected)
        mock_compare["st"]["dataframe"].assert_called_once()
        df_call = mock_compare["st"]["dataframe"].call_args[0][0]
        assert len(df_call) == 4  # truncated to 4


# ===========================================================================
# Tests — metric comparison table
# ===========================================================================


class TestCompareMetricTable:
    def test_dataframe_rendered_for_two_runs(
        self, mock_compare: dict
    ) -> None:
        runs = [
            _make_run("a", result={"metrics": _METRICS}),
            _make_run("b", result={"metrics": _METRICS}),
        ]
        _set_selected(mock_compare, runs)
        compare_page()
        mock_compare["st"]["dataframe"].assert_called_once()

    def test_dataframe_has_expected_columns(
        self, mock_compare: dict
    ) -> None:
        runs = [
            _make_run("a", "SMA", result={"metrics": _METRICS}),
            _make_run("b", "RSI", result={"metrics": _METRICS}),
        ]
        _set_selected(mock_compare, runs)
        compare_page()
        df = mock_compare["st"]["dataframe"].call_args[0][0]
        for col in ("Strategy", "Sharpe", "CAGR", "Max DD", "Win Rate", "Status"):
            assert col in df.columns

    def test_strategy_names_in_dataframe(
        self, mock_compare: dict
    ) -> None:
        runs = [
            _make_run("a", "SMA Crossover", result={"metrics": _METRICS}),
            _make_run("b", "RSI Momentum", result={"metrics": _METRICS}),
        ]
        _set_selected(mock_compare, runs)
        compare_page()
        df = mock_compare["st"]["dataframe"].call_args[0][0]
        assert "SMA Crossover" in df["Strategy"].values
        assert "RSI Momentum" in df["Strategy"].values

    def test_dash_shown_for_run_with_no_result(
        self, mock_compare: dict
    ) -> None:
        runs = [
            _make_run("a", result=None),
            _make_run("b", result=None),
        ]
        _set_selected(mock_compare, runs)
        compare_page()
        df = mock_compare["st"]["dataframe"].call_args[0][0]
        assert "—" in df["Sharpe"].values


# ===========================================================================
# Tests — equity curve overlay
# ===========================================================================


class TestCompareEquityCurves:
    def test_equity_chart_called_when_one_run_has_equity(
        self, mock_compare: dict
    ) -> None:
        runs = [
            _make_run("a", result={"metrics": _METRICS, "equity_curve": _EQUITY_CURVE_A}),
            _make_run("b", result={"metrics": _METRICS}),  # no equity
        ]
        _set_selected(mock_compare, runs)
        compare_page()
        mock_compare["equity_chart"].assert_called_once()

    def test_primary_arg_is_first_equity_run(
        self, mock_compare: dict
    ) -> None:
        runs = [
            _make_run("a", "SMA", result={"metrics": _METRICS, "equity_curve": _EQUITY_CURVE_A}),
            _make_run("b", "RSI", result={"metrics": _METRICS, "equity_curve": _EQUITY_CURVE_B}),
        ]
        _set_selected(mock_compare, runs)
        compare_page()
        primary_arg = mock_compare["equity_chart"].call_args[0][0]
        assert primary_arg == _EQUITY_CURVE_A

    def test_compare_curves_contains_subsequent_equity_runs(
        self, mock_compare: dict
    ) -> None:
        runs = [
            _make_run("a", "SMA", result={"metrics": _METRICS, "equity_curve": _EQUITY_CURVE_A}),
            _make_run("b", "RSI", result={"metrics": _METRICS, "equity_curve": _EQUITY_CURVE_B}),
        ]
        _set_selected(mock_compare, runs)
        compare_page()
        kwargs = mock_compare["equity_chart"].call_args[1]
        compare_curves = kwargs.get("compare_curves")
        assert compare_curves is not None
        assert len(compare_curves) == 1
        assert compare_curves[0][0] == "RSI"
        assert compare_curves[0][1] == _EQUITY_CURVE_B

    def test_single_equity_run_has_no_compare_curves(
        self, mock_compare: dict
    ) -> None:
        runs = [
            _make_run("a", result={"metrics": _METRICS, "equity_curve": _EQUITY_CURVE_A}),
            _make_run("b", result={"metrics": _METRICS}),  # no equity
        ]
        _set_selected(mock_compare, runs)
        compare_page()
        kwargs = mock_compare["equity_chart"].call_args[1]
        compare_curves = kwargs.get("compare_curves")
        assert compare_curves is None

    def test_equity_chart_not_called_when_no_equity_data(
        self, mock_compare: dict
    ) -> None:
        runs = [
            _make_run("a", result={"metrics": _METRICS}),  # no equity_curve
            _make_run("b", result={"metrics": _METRICS}),
        ]
        _set_selected(mock_compare, runs)
        compare_page()
        mock_compare["equity_chart"].assert_not_called()

    def test_caption_shown_when_no_equity_data(
        self, mock_compare: dict
    ) -> None:
        runs = [
            _make_run("a", result={"metrics": _METRICS}),
            _make_run("b", result={"metrics": _METRICS}),
        ]
        _set_selected(mock_compare, runs)
        compare_page()
        caption_texts = [
            str(c[0][0]) for c in mock_compare["st"]["caption"].call_args_list
        ]
        assert any("equity" in t.lower() for t in caption_texts)
