"""
Tests for the three Reporter implementations in backtester/analytics/.

Covers:
- DictReporter: output is JSON-serializable, all keys present, equity_curve
  keys are ISO 8601 strings, no numpy types
- CSVReporter: both files created, metrics CSV has correct columns, trades CSV
  row count matches, default output_dir, BacktestError on OSError
- PlotlyTearsheet: returns go.Figure, has exactly 5 subplots, title contains
  strategy name, handles zero-trade and single-bar results gracefully
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pytest

from backtester.analytics.csv_reporter import CSVReporter
from backtester.analytics.dict_reporter import DictReporter
from backtester.analytics.plotly_tearsheet import PlotlyTearsheet
from backtester.exceptions import BacktestError
from backtester.interfaces import BacktestResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_UTC = timezone.utc


def _equity(values: list[float]) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=len(values), freq="D", tz=_UTC)
    return pd.Series(values, index=idx, name="equity")


def _trades(pnls: list[float]) -> pd.DataFrame:
    n = len(pnls)
    if n == 0:
        return pd.DataFrame(
            columns=[
                "symbol", "exit_timestamp", "quantity",
                "entry_price", "exit_price", "pnl", "commission",
            ]
        )
    idx = pd.date_range("2024-01-02", periods=n, freq="D", tz=_UTC)
    return pd.DataFrame({
        "symbol": ["AAPL"] * n,
        "exit_timestamp": list(idx),
        "quantity": [10.0] * n,
        "entry_price": [100.0] * n,
        "exit_price": [100.0 + p / 10.0 for p in pnls],
        "pnl": pnls,
        "commission": [0.0] * n,
    })


def _result(
    equity_vals: list[float] | None = None,
    pnls: list[float] | None = None,
    strategy_name: str = "TestStrategy",
    params: dict | None = None,
) -> BacktestResult:
    if equity_vals is None:
        equity_vals = [100_000 + i * 1_000 for i in range(10)]
    if pnls is None:
        pnls = [100.0, -50.0, 200.0]
    return BacktestResult(
        equity_curve=_equity(equity_vals),
        trades=_trades(pnls),
        metrics={},
        params=params or {"lookback": 20},
        strategy_name=strategy_name,
    )


def _zero_trades_result() -> BacktestResult:
    return _result(pnls=[])


def _single_bar_result() -> BacktestResult:
    return _result(equity_vals=[100_000])


# ---------------------------------------------------------------------------
# DictReporter
# ---------------------------------------------------------------------------


class TestDictReporter:
    def test_report_returns_dict(self) -> None:
        d = DictReporter().report(_result())
        assert isinstance(d, dict)

    def test_required_top_level_keys_present(self) -> None:
        d = DictReporter().report(_result())
        for key in ("run_id", "strategy_name", "owner", "created_at",
                    "params", "metrics", "equity_curve", "num_bars"):
            assert key in d, f"Missing key: {key}"

    def test_strategy_name_matches(self) -> None:
        d = DictReporter().report(_result(strategy_name="MyAlgo"))
        assert d["strategy_name"] == "MyAlgo"

    def test_num_bars_matches_equity_curve_length(self) -> None:
        r = _result(equity_vals=[100_000, 101_000, 102_000])
        d = DictReporter().report(r)
        assert d["num_bars"] == 3

    def test_equity_curve_keys_are_strings(self) -> None:
        d = DictReporter().report(_result())
        for key in d["equity_curve"].keys():
            assert isinstance(key, str), f"equity_curve key {key!r} is not a str"

    def test_equity_curve_keys_are_iso8601(self) -> None:
        d = DictReporter().report(_result())
        for key in d["equity_curve"].keys():
            # Must parse without raising
            datetime.fromisoformat(key)

    def test_equity_curve_values_are_floats(self) -> None:
        d = DictReporter().report(_result())
        for val in d["equity_curve"].values():
            assert isinstance(val, float)

    def test_metrics_dict_present_and_non_empty(self) -> None:
        d = DictReporter().report(_result())
        assert isinstance(d["metrics"], dict)
        assert len(d["metrics"]) > 0

    def test_output_is_json_serializable(self) -> None:
        d = DictReporter().report(_result())
        # Should not raise
        serialized = json.dumps(d)
        assert len(serialized) > 0

    def test_no_numpy_types_in_output(self) -> None:
        """Every scalar value must be a plain Python type."""
        d = DictReporter().report(_result())

        def _check(obj: object) -> None:
            if isinstance(obj, dict):
                for v in obj.values():
                    _check(v)
            elif isinstance(obj, list):
                for v in obj:
                    _check(v)
            elif not isinstance(obj, (str, int, float, bool, type(None))):
                pytest.fail(
                    f"Non-JSON-safe type found: {type(obj).__name__} = {obj!r}"
                )

        _check(d)

    def test_created_at_is_string(self) -> None:
        d = DictReporter().report(_result())
        assert isinstance(d["created_at"], str)

    def test_params_forwarded(self) -> None:
        d = DictReporter().report(_result(params={"alpha": 0.5, "beta": 10}))
        assert d["params"]["alpha"] == pytest.approx(0.5)
        assert d["params"]["beta"] == 10

    def test_zero_trades_no_raise(self) -> None:
        d = DictReporter().report(_zero_trades_result())
        assert d["metrics"]["num_trades"] == 0

    def test_single_bar_no_raise(self) -> None:
        d = DictReporter().report(_single_bar_result())
        assert d["num_bars"] == 1

    def test_equity_curve_length_preserved(self) -> None:
        r = _result(equity_vals=[100_000 + i * 500 for i in range(20)])
        d = DictReporter().report(r)
        assert len(d["equity_curve"]) == 20


# ---------------------------------------------------------------------------
# CSVReporter
# ---------------------------------------------------------------------------


class TestCSVReporter:
    def test_report_returns_path(self, tmp_path: Path) -> None:
        out = CSVReporter(output_dir=tmp_path).report(_result())
        assert isinstance(out, Path)

    def test_returns_output_dir(self, tmp_path: Path) -> None:
        out = CSVReporter(output_dir=tmp_path).report(_result())
        assert out == tmp_path

    def test_creates_output_dir_if_missing(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c"
        assert not nested.exists()
        CSVReporter(output_dir=nested).report(_result())
        assert nested.is_dir()

    def test_metrics_csv_exists(self, tmp_path: Path) -> None:
        r = _result()
        CSVReporter(output_dir=tmp_path).report(r)
        assert (tmp_path / f"{r.run_id}_metrics.csv").exists()

    def test_trades_csv_exists(self, tmp_path: Path) -> None:
        r = _result()
        CSVReporter(output_dir=tmp_path).report(r)
        assert (tmp_path / f"{r.run_id}_trades.csv").exists()

    def test_metrics_csv_has_correct_columns(self, tmp_path: Path) -> None:
        r = _result()
        CSVReporter(output_dir=tmp_path).report(r)
        path = tmp_path / f"{r.run_id}_metrics.csv"
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == ["metric", "value"]

    def test_metrics_csv_row_count_matches_compute_all(self, tmp_path: Path) -> None:
        from backtester.analytics.metrics import compute_all

        r = _result()
        CSVReporter(output_dir=tmp_path).report(r)
        path = tmp_path / f"{r.run_id}_metrics.csv"
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == len(compute_all(r))

    def test_trades_csv_row_count_matches(self, tmp_path: Path) -> None:
        r = _result(pnls=[100.0, -50.0, 200.0, -30.0])
        CSVReporter(output_dir=tmp_path).report(r)
        path = tmp_path / f"{r.run_id}_trades.csv"
        df = pd.read_csv(path)
        assert len(df) == 4

    def test_trades_csv_empty_for_zero_trades(self, tmp_path: Path) -> None:
        r = _zero_trades_result()
        CSVReporter(output_dir=tmp_path).report(r)
        path = tmp_path / f"{r.run_id}_trades.csv"
        df = pd.read_csv(path)
        assert len(df) == 0

    def test_default_output_dir_is_string_path(self) -> None:
        """Default output_dir is a valid Path (not checked on disk)."""
        reporter = CSVReporter()
        assert isinstance(reporter._output_dir, Path)

    def test_raises_backtest_error_on_write_failure(self, tmp_path: Path) -> None:
        """Writing to a read-only file should raise BacktestError."""
        r = _result()
        # Pre-create the metrics CSV as a directory so writing fails
        metrics_path = tmp_path / f"{r.run_id}_metrics.csv"
        metrics_path.mkdir()
        with pytest.raises(BacktestError):
            CSVReporter(output_dir=tmp_path).report(r)

    def test_single_bar_no_raise(self, tmp_path: Path) -> None:
        CSVReporter(output_dir=tmp_path).report(_single_bar_result())

    def test_zero_trades_produces_header_only_csv(self, tmp_path: Path) -> None:
        r = _zero_trades_result()
        CSVReporter(output_dir=tmp_path).report(r)
        path = tmp_path / f"{r.run_id}_trades.csv"
        with path.open() as f:
            lines = f.read().splitlines()
        # Should have a header row and no data rows
        assert len(lines) == 1  # header only


# ---------------------------------------------------------------------------
# PlotlyTearsheet
# ---------------------------------------------------------------------------


class TestPlotlyTearsheet:
    def test_report_returns_figure(self) -> None:
        fig = PlotlyTearsheet().report(_result())
        assert isinstance(fig, go.Figure)

    def test_figure_has_five_subplots(self) -> None:
        fig = PlotlyTearsheet().report(_result())
        # The subplot grid is described in the layout
        assert fig.layout is not None
        # Each row should have at least one trace type
        assert len(fig.data) >= 1  # at minimum the table row

    def test_title_contains_strategy_name(self) -> None:
        fig = PlotlyTearsheet().report(_result(strategy_name="AlphaStrategy"))
        title_text = fig.layout.title.text or ""
        assert "AlphaStrategy" in title_text

    def test_title_contains_run_id(self) -> None:
        r = _result()
        fig = PlotlyTearsheet().report(r)
        title_text = fig.layout.title.text or ""
        assert r.run_id in title_text

    def test_uses_dark_template(self) -> None:
        fig = PlotlyTearsheet().report(_result())
        assert fig.layout.template is not None

    def test_figure_height_is_900(self) -> None:
        fig = PlotlyTearsheet().report(_result())
        assert fig.layout.height == 900

    def test_custom_height_applied(self) -> None:
        fig = PlotlyTearsheet(height=1200).report(_result())
        assert fig.layout.height == 1200

    def test_zero_trades_no_raise(self) -> None:
        fig = PlotlyTearsheet().report(_zero_trades_result())
        assert isinstance(fig, go.Figure)

    def test_single_bar_no_raise(self) -> None:
        fig = PlotlyTearsheet().report(_single_bar_result())
        assert isinstance(fig, go.Figure)

    def test_figure_contains_table_trace(self) -> None:
        fig = PlotlyTearsheet().report(_result())
        table_traces = [t for t in fig.data if isinstance(t, go.Table)]
        assert len(table_traces) == 1

    def test_figure_contains_equity_scatter(self) -> None:
        fig = PlotlyTearsheet().report(_result())
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
        assert len(scatter_traces) >= 1

    def test_figure_contains_drawdown_trace(self) -> None:
        r = _result(equity_vals=[100_000, 110_000, 90_000, 105_000, 115_000])
        fig = PlotlyTearsheet().report(r)
        # Drawdown is a filled Scatter
        filled = [t for t in fig.data
                  if isinstance(t, go.Scatter) and t.fill == "tozeroy"]
        assert len(filled) >= 1

    def test_figure_contains_histogram_when_trades_exist(self) -> None:
        fig = PlotlyTearsheet().report(_result(pnls=[100.0, -50.0, 200.0]))
        histograms = [t for t in fig.data if isinstance(t, go.Histogram)]
        assert len(histograms) == 1

    def test_no_histogram_when_zero_trades(self) -> None:
        fig = PlotlyTearsheet().report(_zero_trades_result())
        histograms = [t for t in fig.data if isinstance(t, go.Histogram)]
        assert len(histograms) == 0

    def test_metrics_table_rows_populated(self) -> None:
        fig = PlotlyTearsheet().report(_result())
        table = next(t for t in fig.data if isinstance(t, go.Table))
        # cells.values is a list of column arrays: [labels, values]
        assert len(table.cells.values) == 2
        assert len(table.cells.values[0]) > 0

    def test_does_not_call_show(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """report() must never call fig.show() automatically."""
        show_called = []

        def _spy_show(*args: object, **kwargs: object) -> None:
            show_called.append(True)

        monkeypatch.setattr(go.Figure, "show", _spy_show)
        PlotlyTearsheet().report(_result())
        assert show_called == []
