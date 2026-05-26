"""
Tests for ui/components/{run_card, metrics_grid, equity_chart}.

All Streamlit calls are mocked — no live runtime needed.

Covers
------
run_card
    renders without error for a run with a full result dict
    renders without error for a run whose result is None (RUNNING)
    renders without error for DONE and FAILED status runs
    metric pills show "—" when result is None
    metric pills show formatted values when result is present
    clicking the strategy button calls on_click with the run ID

Metric formatting helpers
    fmt_sharpe: 1.234 → "1.23"
    fmt_sharpe: None  → "—"
    fmt_pct:    0.15  → "15.0%"
    fmt_pct:    -0.12 → "-12.0%"
    fmt_pct:    None  → "—"

_detect_format
    "sharpe_ratio" → "ratio"
    "total_return" → "percent"
    "cagr"         → "percent"
    "max_drawdown" → "percent"
    "win_rate"     → "percent"
    "num_trades"   → "integer"
    "unknown_key"  → "ratio"  (fallback)

_fmt_value
    percent format: 0.18 → "18.0%"
    ratio format:   1.42 → "1.42"
    integer format: 42.7 → "43"
    None value      → "—"

metrics_grid
    calls st.metric exactly N times for an N-key dict
    uses correct number of columns

metric_card
    calls st.metric with formatted label, value, delta

equity_chart
    calls st.plotly_chart exactly once
    passes a plotly Figure object
    adds an extra trace per compare_curves entry
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Callable
from unittest.mock import MagicMock, patch, call
import uuid

import pytest

from ui.components.metrics_grid import (
    _detect_format,
    _fmt_value,
    metric_card,
    metrics_grid,
)
from ui.components.run_card import fmt_sharpe, fmt_pct, run_card, time_ago
from ui.components.equity_chart import equity_chart


# ===========================================================================
# Fixtures & factories
# ===========================================================================


def _make_run(
    *,
    result: dict | None = None,
    status: str = "DONE",
    visibility: str = "TEAM",
    owner_id: int = 1,
    tags: list | None = None,
) -> MagicMock:
    """Return a mock Run-like object."""
    run = MagicMock()
    run.id = str(uuid.uuid4())
    run.strategy_name = "TestStrategy"
    run.status = status
    run.visibility = visibility
    run.owner_id = owner_id
    run.result = result
    run.tags = tags or []
    run.created_at = datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc)
    return run


_FULL_RESULT = {
    "metrics": {
        "sharpe_ratio": 1.234,
        "cagr": 0.15,
        "max_drawdown": -0.12,
        "win_rate": 0.55,
    },
    "equity_curve": {"2024-01-01": 10000, "2024-12-31": 11500},
}

_CURRENT_USER = {
    "user_id": 1,
    "name": "Alice Analyst",
    "email": "alice@example.com",
    "role": "analyst",
    "avatar_initials": "AA",
}


def _make_col_mock() -> MagicMock:
    col = MagicMock()
    col.__enter__ = MagicMock(return_value=col)
    col.__exit__ = MagicMock(return_value=False)
    # Allow attribute metric to be a MagicMock as well
    col.metric = MagicMock()
    return col


def _columns_side_effect(spec):
    """Return the right number of column mocks based on spec."""
    if isinstance(spec, (list, tuple)):
        n = len(spec)
    else:
        n = int(spec)
    return [_make_col_mock() for _ in range(n)]


@pytest.fixture
def mock_st_run_card(monkeypatch):
    """Mock all st.* calls used by run_card."""
    mock_container = MagicMock()
    mock_container.__enter__ = MagicMock(return_value=mock_container)
    mock_container.__exit__ = MagicMock(return_value=False)

    button_mock = MagicMock(return_value=False)

    monkeypatch.setattr("ui.components.run_card.st.container", MagicMock(return_value=mock_container))
    monkeypatch.setattr("ui.components.run_card.st.columns", MagicMock(side_effect=_columns_side_effect))
    monkeypatch.setattr("ui.components.run_card.st.button", button_mock)
    monkeypatch.setattr("ui.components.run_card.st.metric", MagicMock())
    monkeypatch.setattr("ui.components.run_card.st.markdown", MagicMock())

    # Mock get_db so run_card doesn't hit a real DB when owner != current_user
    @contextmanager
    def _mock_get_db():
        mock_db = MagicMock()
        yield mock_db

    monkeypatch.setattr("ui.components.run_card.get_db", _mock_get_db)
    monkeypatch.setattr(
        "ui.components.run_card.get_user_by_id",
        MagicMock(return_value=None),
    )

    return {"button": button_mock, "container": mock_container}


@pytest.fixture
def mock_st_metrics(monkeypatch):
    """Mock st.* calls used by metrics_grid / metric_card."""
    metric_mock = MagicMock()
    cols_mock = [MagicMock() for _ in range(4)]
    for col in cols_mock:
        col.__enter__ = MagicMock(return_value=col)
        col.__exit__ = MagicMock(return_value=False)

    monkeypatch.setattr("ui.components.metrics_grid.st.metric", metric_mock)
    monkeypatch.setattr("ui.components.metrics_grid.st.columns", MagicMock(return_value=cols_mock))
    monkeypatch.setattr("ui.components.metrics_grid.st.markdown", MagicMock())
    monkeypatch.setattr("ui.components.metrics_grid.st.info", MagicMock())
    return {"metric": metric_mock}


@pytest.fixture
def mock_st_chart(monkeypatch):
    """Mock st.plotly_chart used by equity_chart."""
    plotly_mock = MagicMock()
    monkeypatch.setattr("ui.components.equity_chart.st.plotly_chart", plotly_mock)
    return {"plotly_chart": plotly_mock}


# ===========================================================================
# Tests — fmt_sharpe / fmt_pct (pure helpers in run_card)
# ===========================================================================


class TestMetricFormatHelpers:
    def test_fmt_sharpe_formats_to_two_dp(self) -> None:
        assert fmt_sharpe(1.234) == "1.23"

    def test_fmt_sharpe_none_returns_dash(self) -> None:
        assert fmt_sharpe(None) == "—"

    def test_fmt_sharpe_negative(self) -> None:
        assert fmt_sharpe(-0.5) == "-0.50"

    def test_fmt_pct_positive(self) -> None:
        assert fmt_pct(0.15) == "15.0%"

    def test_fmt_pct_negative(self) -> None:
        assert fmt_pct(-0.12) == "-12.0%"

    def test_fmt_pct_none_returns_dash(self) -> None:
        assert fmt_pct(None) == "—"

    def test_fmt_pct_zero(self) -> None:
        assert fmt_pct(0.0) == "0.0%"


# ===========================================================================
# Tests — _detect_format
# ===========================================================================


class TestDetectFormat:
    def test_sharpe_ratio_is_ratio(self) -> None:
        assert _detect_format("sharpe_ratio") == "ratio"

    def test_sortino_ratio_is_ratio(self) -> None:
        assert _detect_format("sortino_ratio") == "ratio"

    def test_calmar_is_ratio(self) -> None:
        assert _detect_format("calmar") == "ratio"

    def test_total_return_is_percent(self) -> None:
        assert _detect_format("total_return") == "percent"

    def test_cagr_is_percent(self) -> None:
        assert _detect_format("cagr") == "percent"

    def test_max_drawdown_is_percent(self) -> None:
        assert _detect_format("max_drawdown") == "percent"

    def test_win_rate_is_percent(self) -> None:
        assert _detect_format("win_rate") == "percent"

    def test_num_trades_is_integer(self) -> None:
        assert _detect_format("num_trades") == "integer"

    def test_trade_count_is_integer(self) -> None:
        assert _detect_format("trade_count") == "integer"

    def test_unknown_key_falls_back_to_ratio(self) -> None:
        assert _detect_format("unknown_metric_xyz") == "ratio"

    def test_case_insensitive(self) -> None:
        assert _detect_format("SHARPE") == "ratio"
        assert _detect_format("CAGR") == "percent"


# ===========================================================================
# Tests — _fmt_value
# ===========================================================================


class TestFmtValue:
    def test_percent_format(self) -> None:
        assert _fmt_value(0.18, "percent") == "18.0%"

    def test_ratio_format(self) -> None:
        assert _fmt_value(1.42, "ratio") == "1.42"

    def test_integer_format_rounds(self) -> None:
        assert _fmt_value(42.7, "integer") == "43"

    def test_integer_format_exact(self) -> None:
        assert _fmt_value(10.0, "integer") == "10"

    def test_none_returns_dash(self) -> None:
        assert _fmt_value(None, "ratio") == "—"

    def test_currency_format(self) -> None:
        result = _fmt_value(1234.5, "currency")
        assert "$" in result
        assert "1,234" in result

    def test_negative_percent(self) -> None:
        assert _fmt_value(-0.05, "percent") == "-5.0%"


# ===========================================================================
# Tests — run_card rendering
# ===========================================================================


class TestRunCard:
    def test_renders_run_with_full_result(
        self, mock_st_run_card: dict
    ) -> None:
        """run_card must not raise for a run with a complete result dict."""
        run = _make_run(result=_FULL_RESULT)
        on_click = MagicMock()
        run_card(run, _CURRENT_USER, on_click)  # no exception

    def test_renders_run_with_no_result(
        self, mock_st_run_card: dict
    ) -> None:
        """run_card must not raise when run.result is None (RUNNING state)."""
        run = _make_run(result=None, status="RUNNING")
        on_click = MagicMock()
        run_card(run, _CURRENT_USER, on_click)

    def test_renders_done_status(self, mock_st_run_card: dict) -> None:
        run = _make_run(result=_FULL_RESULT, status="DONE")
        run_card(run, _CURRENT_USER, MagicMock())

    def test_renders_failed_status(self, mock_st_run_card: dict) -> None:
        run = _make_run(result=None, status="FAILED")
        run_card(run, _CURRENT_USER, MagicMock())

    def test_strategy_button_key_contains_run_id(
        self, mock_st_run_card: dict
    ) -> None:
        """The strategy-name button must use key='run_card_{run.id}'."""
        run = _make_run(result=_FULL_RESULT)
        run_card(run, _CURRENT_USER, MagicMock())
        button_calls = mock_st_run_card["button"].call_args_list
        keys_used = [c[1].get("key", "") for c in button_calls]
        assert any(run.id in k for k in keys_used)

    def test_clicking_button_calls_on_click(
        self, mock_st_run_card: dict, monkeypatch
    ) -> None:
        """When the strategy button is pressed, on_click(run.id) is called."""
        run = _make_run(result=_FULL_RESULT)
        on_click = MagicMock()

        # Make the button return True (clicked) for the run's card key
        def _clicked(label, key="", **kwargs):
            return key == f"run_card_{run.id}"

        mock_st_run_card["button"].side_effect = _clicked
        run_card(run, _CURRENT_USER, on_click)
        on_click.assert_called_once_with(run.id)

    def test_no_click_means_on_click_not_called(
        self, mock_st_run_card: dict
    ) -> None:
        run = _make_run(result=_FULL_RESULT)
        on_click = MagicMock()
        # button always returns False
        run_card(run, _CURRENT_USER, on_click)
        on_click.assert_not_called()

    def test_run_with_tags(self, mock_st_run_card: dict) -> None:
        """Tags should not cause an error."""
        run = _make_run(result=_FULL_RESULT, tags=["momentum", "daily"])
        run_card(run, _CURRENT_USER, MagicMock())

    def test_other_owner_triggers_db_lookup(
        self, mock_st_run_card: dict, monkeypatch
    ) -> None:
        """When run.owner_id != current_user, get_user_by_id is called."""
        mock_lookup = MagicMock(return_value=None)
        monkeypatch.setattr("ui.components.run_card.get_user_by_id", mock_lookup)

        run = _make_run(result=_FULL_RESULT, owner_id=999)  # different user
        run_card(run, _CURRENT_USER, MagicMock())
        mock_lookup.assert_called_once()


# ===========================================================================
# Tests — metrics_grid
# ===========================================================================


class TestMetricsGrid:
    def test_calls_metric_n_times_for_n_keys(
        self, mock_st_metrics: dict
    ) -> None:
        data = {
            "sharpe_ratio": 1.42,
            "cagr": 0.18,
            "max_drawdown": -0.11,
            "win_rate": 0.55,
            "num_trades": 120,
            "total_return": 0.25,
        }
        metrics_grid(data, columns=3)
        assert mock_st_metrics["metric"].call_count == 6

    def test_empty_dict_shows_info(self, mock_st_metrics: dict, monkeypatch) -> None:
        info_mock = MagicMock()
        monkeypatch.setattr("ui.components.metrics_grid.st.info", info_mock)
        metrics_grid({}, columns=4)
        info_mock.assert_called_once()
        mock_st_metrics["metric"].assert_not_called()

    def test_columns_count_passed_to_st_columns(
        self, mock_st_metrics: dict, monkeypatch
    ) -> None:
        cols_mock = MagicMock()
        returned_cols = [MagicMock() for _ in range(3)]
        for col in returned_cols:
            col.__enter__ = MagicMock(return_value=col)
            col.__exit__ = MagicMock(return_value=False)
        cols_mock.return_value = returned_cols
        monkeypatch.setattr("ui.components.metrics_grid.st.columns", cols_mock)

        metrics_grid({"a": 1, "b": 2}, columns=3)
        cols_mock.assert_called_with(3)


# ===========================================================================
# Tests — metric_card
# ===========================================================================


class TestMetricCard:
    def test_calls_st_metric_once(self, mock_st_metrics: dict) -> None:
        metric_card("sharpe_ratio", 1.42)
        mock_st_metrics["metric"].assert_called_once()

    def test_label_is_title_cased(self, mock_st_metrics: dict) -> None:
        metric_card("sharpe_ratio", 1.42)
        _, kwargs = mock_st_metrics["metric"].call_args
        assert kwargs["label"] == "Sharpe Ratio"

    def test_value_formatted_as_ratio(self, mock_st_metrics: dict) -> None:
        metric_card("sharpe_ratio", 1.5)
        _, kwargs = mock_st_metrics["metric"].call_args
        assert kwargs["value"] == "1.50"

    def test_value_formatted_as_percent(self, mock_st_metrics: dict) -> None:
        metric_card("cagr", 0.18)
        _, kwargs = mock_st_metrics["metric"].call_args
        assert kwargs["value"] == "18.0%"

    def test_delta_formatted_and_passed(self, mock_st_metrics: dict) -> None:
        metric_card("cagr", 0.18, delta=0.03)
        _, kwargs = mock_st_metrics["metric"].call_args
        assert kwargs["delta"] == "3.0%"

    def test_delta_none_passes_none(self, mock_st_metrics: dict) -> None:
        metric_card("sharpe_ratio", 1.42, delta=None)
        _, kwargs = mock_st_metrics["metric"].call_args
        assert kwargs["delta"] is None

    def test_explicit_format_overrides_auto(self, mock_st_metrics: dict) -> None:
        # "sharpe_ratio" would auto-detect as ratio, but we force percent
        metric_card("sharpe_ratio", 0.5, format="percent")
        _, kwargs = mock_st_metrics["metric"].call_args
        assert kwargs["value"] == "50.0%"


# ===========================================================================
# Tests — equity_chart
# ===========================================================================


class TestEquityChart:
    def test_calls_plotly_chart_once(self, mock_st_chart: dict) -> None:
        curve = {"2024-01-01": 10000, "2024-06-01": 11000, "2024-12-31": 12000}
        equity_chart(curve)
        mock_st_chart["plotly_chart"].assert_called_once()

    def test_passes_figure_object(self, mock_st_chart: dict) -> None:
        import plotly.graph_objects as go
        curve = {"2024-01-01": 10000, "2024-12-31": 11500}
        equity_chart(curve)
        args, kwargs = mock_st_chart["plotly_chart"].call_args
        fig = args[0]
        assert isinstance(fig, go.Figure)

    def test_use_container_width_true(self, mock_st_chart: dict) -> None:
        curve = {"2024-01-01": 10000}
        equity_chart(curve)
        _, kwargs = mock_st_chart["plotly_chart"].call_args
        assert kwargs.get("use_container_width") is True

    def test_primary_trace_is_added(self, mock_st_chart: dict) -> None:
        import plotly.graph_objects as go
        curve = {"2024-01-01": 10000, "2024-12-31": 11500}
        equity_chart(curve, title="My Curve")
        args, _ = mock_st_chart["plotly_chart"].call_args
        fig: go.Figure = args[0]
        assert len(fig.data) >= 1

    def test_compare_curves_add_extra_traces(
        self, mock_st_chart: dict
    ) -> None:
        import plotly.graph_objects as go
        primary = {"2024-01-01": 10000, "2024-12-31": 11500}
        compare = [
            ("Benchmark A", {"2024-01-01": 10000, "2024-12-31": 10800}),
            ("Benchmark B", {"2024-01-01": 10000, "2024-12-31": 10500}),
        ]
        equity_chart(primary, compare_curves=compare)
        args, _ = mock_st_chart["plotly_chart"].call_args
        fig: go.Figure = args[0]
        # Primary + 2 compare traces (+ possibly a reference line shape)
        assert len(fig.data) == 3

    def test_accepts_pandas_series(self, mock_st_chart: dict) -> None:
        import pandas as pd
        idx = pd.date_range("2024-01-01", periods=12, freq="ME")
        series = pd.Series(range(10000, 11200, 100), index=idx)
        equity_chart(series)  # should not raise
        mock_st_chart["plotly_chart"].assert_called_once()

    def test_custom_height_passed_to_layout(
        self, mock_st_chart: dict
    ) -> None:
        import plotly.graph_objects as go
        curve = {"2024-01-01": 10000}
        equity_chart(curve, height=500)
        args, _ = mock_st_chart["plotly_chart"].call_args
        fig: go.Figure = args[0]
        assert fig.layout.height == 500
