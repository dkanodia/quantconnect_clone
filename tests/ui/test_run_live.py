"""
Tests for ui.pages.run_live.

Covers pure-computation helpers (no Streamlit) and page-level guards.
The animation itself is entirely client-side Plotly JS — not tested here.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_ohlcv() -> pd.DataFrame:
    """100 business-day OHLCV DataFrame with a deterministic random walk."""
    dates = pd.date_range("2022-01-03", periods=100, freq="B")
    rng   = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.standard_normal(100))
    return pd.DataFrame(
        {
            "Open":  close - 0.5,
            "High":  close + 1.0,
            "Low":   close - 1.0,
            "Close": close,
        },
        index=dates,
    )


@pytest.fixture()
def st_mocks(monkeypatch):
    import streamlit as st

    mocks = {
        "title":        MagicMock(),
        "caption":      MagicMock(),
        "subheader":    MagicMock(),
        "warning":      MagicMock(),
        "error":        MagicMock(),
        "success":      MagicMock(),
        "info":         MagicMock(),
        "markdown":     MagicMock(),
        "metric":       MagicMock(),
        "plotly_chart": MagicMock(),
        "button":       MagicMock(return_value=False),
        "spinner":      MagicMock(
            __enter__=MagicMock(return_value=None),
            __exit__=MagicMock(return_value=False),
        ),
        "columns":      MagicMock(
            return_value=[MagicMock(
                __enter__=MagicMock(return_value=MagicMock()),
                __exit__=MagicMock(return_value=False),
            ) for _ in range(4)]
        ),
        "rerun": MagicMock(side_effect=Exception("rerun")),
        "stop":  MagicMock(side_effect=Exception("stop")),
    }
    for name, mock in mocks.items():
        monkeypatch.setattr(st, name, mock)
    return mocks


@pytest.fixture()
def mock_auth(monkeypatch):
    monkeypatch.setattr(
        "ui.pages.run_live.get_current_user",
        lambda: {"user_id": 1, "role": "analyst", "name": "Tester"},
    )
    monkeypatch.setattr("ui.pages.run_live.require_role", MagicMock())


# ---------------------------------------------------------------------------
# Helper: build a fake get_db context manager
# ---------------------------------------------------------------------------


def _fake_db_cm():
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=MagicMock())
    cm.__exit__  = MagicMock(return_value=False)
    return cm


# ---------------------------------------------------------------------------
# Tests — SMA crossover
# ---------------------------------------------------------------------------


class TestSMACrossover:
    def test_returns_two_series(self, fake_ohlcv):
        from ui.pages.run_live import _sma_crossover

        signal, sma = _sma_crossover(fake_ohlcv["Close"])
        assert isinstance(signal, pd.Series)
        assert isinstance(sma, pd.Series)
        assert len(signal) == len(fake_ohlcv)

    def test_signal_is_binary(self, fake_ohlcv):
        from ui.pages.run_live import _sma_crossover

        signal, _ = _sma_crossover(fake_ohlcv["Close"])
        assert signal.isin([0.0, 1.0]).all()

    def test_first_window_bars_are_zero(self, fake_ohlcv):
        from ui.pages.run_live import _sma_crossover, _SMA_WINDOW

        signal, _ = _sma_crossover(fake_ohlcv["Close"])
        assert (signal.iloc[:_SMA_WINDOW] == 0.0).all()

    def test_custom_window(self, fake_ohlcv):
        from ui.pages.run_live import _sma_crossover

        _, sma = _sma_crossover(fake_ohlcv["Close"], window=5)
        assert sma.iloc[:4].isna().all()
        assert not sma.iloc[5:].isna().any()


# ---------------------------------------------------------------------------
# Tests — equity computation
# ---------------------------------------------------------------------------


class TestComputeEquity:
    def test_starts_at_initial_cash(self, fake_ohlcv):
        from ui.pages.run_live import _compute_equity, _sma_crossover

        signal, _ = _sma_crossover(fake_ohlcv["Close"])
        equity = _compute_equity(fake_ohlcv["Close"], signal, 50_000)
        assert equity.iloc[0] == 50_000

    def test_length_matches_input(self, fake_ohlcv):
        from ui.pages.run_live import _compute_equity, _sma_crossover

        signal, _ = _sma_crossover(fake_ohlcv["Close"])
        equity = _compute_equity(fake_ohlcv["Close"], signal, 100_000)
        assert len(equity) == len(fake_ohlcv)

    def test_equity_is_positive(self, fake_ohlcv):
        from ui.pages.run_live import _compute_equity, _sma_crossover

        signal, _ = _sma_crossover(fake_ohlcv["Close"])
        equity = _compute_equity(fake_ohlcv["Close"], signal, 100_000)
        assert (equity > 0).all()

    def test_zero_signal_means_flat(self, fake_ohlcv):
        from ui.pages.run_live import _compute_equity

        zero = pd.Series(0.0, index=fake_ohlcv.index)
        equity = _compute_equity(fake_ohlcv["Close"], zero, 100_000)
        assert (equity == 100_000).all()


# ---------------------------------------------------------------------------
# Tests — metrics
# ---------------------------------------------------------------------------


class TestBuildMetrics:
    def _equity_and_signal(self, fake_ohlcv, cash=100_000):
        from ui.pages.run_live import _compute_equity, _sma_crossover

        signal, _ = _sma_crossover(fake_ohlcv["Close"])
        return _compute_equity(fake_ohlcv["Close"], signal, cash), signal

    def test_all_required_keys(self, fake_ohlcv):
        from ui.pages.run_live import _build_metrics

        equity, signal = self._equity_and_signal(fake_ohlcv)
        m = _build_metrics(equity, signal, 100_000)
        for key in ("total_return", "cagr", "sharpe_ratio", "max_drawdown", "win_rate", "total_trades"):
            assert key in m

    def test_max_drawdown_non_positive(self, fake_ohlcv):
        from ui.pages.run_live import _build_metrics

        equity, signal = self._equity_and_signal(fake_ohlcv)
        assert _build_metrics(equity, signal, 100_000)["max_drawdown"] <= 0.0

    def test_win_rate_in_unit_interval(self, fake_ohlcv):
        from ui.pages.run_live import _build_metrics

        equity, signal = self._equity_and_signal(fake_ohlcv)
        wr = _build_metrics(equity, signal, 100_000)["win_rate"]
        assert 0.0 <= wr <= 1.0

    def test_total_trades_non_negative_int(self, fake_ohlcv):
        from ui.pages.run_live import _build_metrics

        equity, signal = self._equity_and_signal(fake_ohlcv)
        trades = _build_metrics(equity, signal, 100_000)["total_trades"]
        assert isinstance(trades, int) and trades >= 0

    def test_total_return_close_to_equity_endpoints(self, fake_ohlcv):
        from ui.pages.run_live import _build_metrics

        cash = 100_000.0
        equity, signal = self._equity_and_signal(fake_ohlcv, cash)
        m = _build_metrics(equity, signal, cash)
        expected = (equity.iloc[-1] - cash) / cash
        assert abs(m["total_return"] - expected) < 1e-4  # rounded to 4dp


# ---------------------------------------------------------------------------
# Tests — animated figure builder
# ---------------------------------------------------------------------------


class TestBuildAnimatedFigure:
    def test_returns_go_figure(self, fake_ohlcv):
        from ui.pages.run_live import (
            _build_animated_figure,
            _compute_equity,
            _sma_crossover,
        )

        signal, sma = _sma_crossover(fake_ohlcv["Close"])
        equity = _compute_equity(fake_ohlcv["Close"], signal, 100_000)
        fig = _build_animated_figure(fake_ohlcv, sma, equity, "AAPL", 100_000)
        assert isinstance(fig, go.Figure)

    def test_figure_has_frames(self, fake_ohlcv):
        from ui.pages.run_live import (
            _build_animated_figure,
            _compute_equity,
            _sma_crossover,
        )

        signal, sma = _sma_crossover(fake_ohlcv["Close"])
        equity = _compute_equity(fake_ohlcv["Close"], signal, 100_000)
        fig = _build_animated_figure(fake_ohlcv, sma, equity, "SPY", 100_000)
        assert len(fig.frames) > 0

    def test_figure_has_four_traces(self, fake_ohlcv):
        """candle + SMA + reference_line + equity = 4 initial traces."""
        from ui.pages.run_live import (
            _build_animated_figure,
            _compute_equity,
            _sma_crossover,
        )

        signal, sma = _sma_crossover(fake_ohlcv["Close"])
        equity = _compute_equity(fake_ohlcv["Close"], signal, 100_000)
        fig = _build_animated_figure(fake_ohlcv, sma, equity, "SPY", 100_000)
        assert len(fig.data) == 4

    def test_frames_update_three_traces(self, fake_ohlcv):
        """Each frame should update traces 0, 1, 3 (not 2 — the static line)."""
        from ui.pages.run_live import (
            _build_animated_figure,
            _compute_equity,
            _sma_crossover,
        )

        signal, sma = _sma_crossover(fake_ohlcv["Close"])
        equity = _compute_equity(fake_ohlcv["Close"], signal, 100_000)
        fig = _build_animated_figure(fake_ohlcv, sma, equity, "SPY", 100_000)
        for frame in fig.frames:
            assert list(frame.traces) == [0, 1, 3]

    def test_has_play_button(self, fake_ohlcv):
        from ui.pages.run_live import (
            _build_animated_figure,
            _compute_equity,
            _sma_crossover,
        )

        signal, sma = _sma_crossover(fake_ohlcv["Close"])
        equity = _compute_equity(fake_ohlcv["Close"], signal, 100_000)
        fig = _build_animated_figure(fake_ohlcv, sma, equity, "SPY", 100_000)
        menus = fig.layout.updatemenus
        assert len(menus) == 1
        labels = [b.label for b in menus[0].buttons]
        assert any("Play" in lbl for lbl in labels)


# ---------------------------------------------------------------------------
# Tests — page-level guards
# ---------------------------------------------------------------------------

import plotly.graph_objects as go  # noqa: E402  (needed for isinstance check above)


class TestRunLivePageGuards:
    def test_no_run_id_shows_warning(self, monkeypatch, st_mocks, mock_auth):
        import streamlit as st

        monkeypatch.setattr(st, "session_state", {"selected_run_id": None})
        from ui.pages.run_live import run_live_page

        run_live_page()
        st.warning.assert_called_once()

    def test_run_not_found_shows_error(self, monkeypatch, st_mocks, mock_auth):
        import streamlit as st

        monkeypatch.setattr(st, "session_state", {"selected_run_id": "bad-id"})
        monkeypatch.setattr("ui.pages.run_live.get_run", lambda *_: None)
        monkeypatch.setattr("ui.pages.run_live.get_db", lambda: _fake_db_cm())

        from ui.pages.run_live import run_live_page

        run_live_page()
        st.error.assert_called_once()

    def test_done_run_redirects(self, monkeypatch, st_mocks, mock_auth):
        import streamlit as st

        session = {"selected_run_id": "uid-1", "page": "run_live"}
        monkeypatch.setattr(st, "session_state", session)

        fake_run = MagicMock()
        fake_run.status = "DONE"
        monkeypatch.setattr("ui.pages.run_live.get_run", lambda *_: fake_run)
        monkeypatch.setattr("ui.pages.run_live.get_db", lambda: _fake_db_cm())

        from ui.pages.run_live import run_live_page

        with pytest.raises(Exception, match="rerun"):
            run_live_page()
        assert session["page"] == "run_detail"

    def test_failed_run_redirects(self, monkeypatch, st_mocks, mock_auth):
        import streamlit as st

        session = {"selected_run_id": "uid-2", "page": "run_live"}
        monkeypatch.setattr(st, "session_state", session)

        fake_run = MagicMock()
        fake_run.status = "FAILED"
        monkeypatch.setattr("ui.pages.run_live.get_run", lambda *_: fake_run)
        monkeypatch.setattr("ui.pages.run_live.get_db", lambda: _fake_db_cm())

        from ui.pages.run_live import run_live_page

        with pytest.raises(Exception, match="rerun"):
            run_live_page()
        assert session["page"] == "run_detail"

    def test_empty_ohlcv_marks_failed(self, monkeypatch, st_mocks, mock_auth):
        import streamlit as st
        from ui.pages import run_live as rl

        session = {"selected_run_id": "uid-3"}
        monkeypatch.setattr(st, "session_state", session)

        fake_run = MagicMock()
        fake_run.status        = "RUNNING"
        fake_run.params        = {"symbols": ["FAKE"], "start_date": "2022-01-01",
                                  "end_date": "2022-12-31", "initial_capital": 100_000}
        fake_run.strategy_name = "TestStrat"

        mock_update = MagicMock()
        monkeypatch.setattr("ui.pages.run_live.get_run",           lambda *_: fake_run)
        monkeypatch.setattr("ui.pages.run_live.get_db",            lambda: _fake_db_cm())
        monkeypatch.setattr("ui.pages.run_live._fetch_ohlcv",      lambda *_: pd.DataFrame())
        monkeypatch.setattr("ui.pages.run_live.update_run_status",  mock_update)

        rl.run_live_page()

        mock_update.assert_called_once()
        assert "FAILED" in mock_update.call_args.args
