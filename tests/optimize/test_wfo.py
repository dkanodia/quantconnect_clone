"""
Tests for backtester/optimize/wfo.py (WFOOrchestrator).

Covers:
- Rolling and anchored modes both produce n_windows summaries
- oos_equity_curve length equals the sum of OOS bar counts across windows
- decay_scores length equals n_windows; is/oos sharpe lists too
- robustness_score is a finite, non-negative float
- Stitched OOS equity curve is monotonically and uniquely indexed
- ListFeed iteration/reset exercised indirectly (multi-trial IS optimisation
  re-iterates the same window feed; OOS curve lengths match bar counts)
- Configuration validation (bad window_type, n_windows, is_pct, no factory)
- WFOError when the feed is too small to partition
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator

import pandas as pd
import pytest

from backtester.core.backtester import Backtester
from backtester.exceptions import ConfigurationError, WFOError
from backtester.execution.commission_models import ZeroCommission
from backtester.execution.execution_models import NextOpenExecution
from backtester.execution.slippage_models import ZeroSlippage
from backtester.interfaces import Bar, BacktestResult, DataFeed, Reporter, Strategy
from backtester.optimize.grid_search import GridSearch
from backtester.optimize.param_space import IntParam, ParamSpace
from backtester.optimize.wfo import WFOOrchestrator, WFOResult, WFOWindowSummary
from backtester.risk.risk_models import NoRisk
from backtester.strategy.event_driven import EventDrivenStrategy

_UTC = timezone.utc


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _SeqFeed(DataFeed):
    """Test-local DataFeed; the WFO replaces feeds with its internal ListFeed."""

    def __init__(self, bars: list[Bar]) -> None:
        self._bars = list(bars)

    def __iter__(self) -> Iterator[Bar]:
        return iter(self._bars)

    def reset(self) -> None:
        pass

    @property
    def symbols(self) -> list[str]:
        return sorted({b.symbol for b in self._bars})


class _NoopReporter(Reporter):
    def report(self, result: BacktestResult) -> None:
        return None


class EntryDelayStrategy(EventDrivenStrategy):
    """Buys 10 shares on the ``lookback``-th bar, then holds for the window."""

    def __init__(self, lookback: int = 1) -> None:
        super().__init__(lookback=lookback)
        self.lookback = lookback
        self._seen = 0
        self._bought = False

    def on_bar(self, bar: Bar, context: Any = None) -> None:
        self._seen += 1
        if not self._bought and self._seen >= self.lookback:
            self.buy(bar.symbol, quantity=10)
            self._bought = True


def _noisy_uptrend(n: int = 200, symbol: str = "AAPL", seed: int = 42) -> list[Bar]:
    """Rising series with noise so equity returns vary (non-zero Sharpe)."""
    rng = random.Random(seed)
    base = datetime(2024, 1, 1, tzinfo=_UTC)
    bars: list[Bar] = []
    price = 100.0
    for i in range(n):
        price = max(1.0, price + 0.5 + rng.uniform(-0.8, 0.8))
        bars.append(
            Bar(symbol, base + timedelta(days=i), price, price + 0.4,
                price - 0.4, price, 1_000.0)
        )
    return bars


def _factory():
    """Build a backtester_factory whose components the WFO reuses per window."""
    def factory(strategy: Strategy) -> Backtester:
        slip = ZeroSlippage()
        comm = ZeroCommission()
        return Backtester(
            feed=_SeqFeed([]),  # placeholder; WFO supplies the real feed
            strategy=strategy,
            execution=NextOpenExecution(slip, comm),
            slippage=slip,
            commission=comm,
            risk=NoRisk(),
            reporter=_NoopReporter(),
            initial_cash=100_000.0,
        )
    return factory


def _total_return(result: BacktestResult) -> float:
    return float(result.metrics["total_return"])


def _make_orchestrator(window_type: str = "rolling", n_windows: int = 4) -> WFOOrchestrator:
    space = ParamSpace([IntParam("lookback", 1, 3)])
    return WFOOrchestrator(
        strategy_class=EntryDelayStrategy,
        param_space=space,
        optimizer=GridSearch(),
        objective=_total_return,
        n_trials=10,            # >= grid size → exhaustive, deterministic
        window_type=window_type,
        n_windows=n_windows,
        is_pct=0.7,
        backtester_factory=_factory(),
    )


# ---------------------------------------------------------------------------
# Construction / configuration
# ---------------------------------------------------------------------------


class TestConstructionValidation:
    def test_bad_window_type_raises(self) -> None:
        with pytest.raises(ConfigurationError):
            _make_orchestrator(window_type="diagonal")

    def test_zero_windows_raises(self) -> None:
        with pytest.raises(ConfigurationError):
            WFOOrchestrator(
                EntryDelayStrategy, ParamSpace([IntParam("lookback", 1, 3)]),
                GridSearch(), _total_return, n_windows=0,
                backtester_factory=_factory(),
            )

    def test_is_pct_out_of_range_raises(self) -> None:
        with pytest.raises(ConfigurationError):
            WFOOrchestrator(
                EntryDelayStrategy, ParamSpace([IntParam("lookback", 1, 3)]),
                GridSearch(), _total_return, is_pct=1.5,
                backtester_factory=_factory(),
            )

    def test_no_factory_raises_on_run(self) -> None:
        orch = WFOOrchestrator(
            EntryDelayStrategy, ParamSpace([IntParam("lookback", 1, 3)]),
            GridSearch(), _total_return, n_windows=4,
            backtester_factory=None,
        )
        with pytest.raises(ConfigurationError):
            orch.run(_SeqFeed(_noisy_uptrend(200)))

    def test_empty_feed_raises_wfo_error(self) -> None:
        with pytest.raises(WFOError):
            _make_orchestrator().run(_SeqFeed([]))

    def test_too_few_bars_raises_wfo_error(self) -> None:
        # 4 windows need >= 8 bars; give it 6.
        with pytest.raises(WFOError):
            _make_orchestrator(n_windows=4).run(_SeqFeed(_noisy_uptrend(6)))


# ---------------------------------------------------------------------------
# Rolling / anchored window counts
# ---------------------------------------------------------------------------


class TestWindowCounts:
    @pytest.mark.parametrize("mode", ["rolling", "anchored"])
    def test_produces_n_windows_summaries(self, mode: str) -> None:
        result = _make_orchestrator(window_type=mode, n_windows=4).run(
            _SeqFeed(_noisy_uptrend(200))
        )
        assert isinstance(result, WFOResult)
        assert len(result.windows) == 4
        assert all(isinstance(w, WFOWindowSummary) for w in result.windows)

    @pytest.mark.parametrize("mode", ["rolling", "anchored"])
    def test_window_indices_are_sequential(self, mode: str) -> None:
        result = _make_orchestrator(window_type=mode, n_windows=5).run(
            _SeqFeed(_noisy_uptrend(250))
        )
        assert [w.window_index for w in result.windows] == [0, 1, 2, 3, 4]

    def test_anchored_is_bars_non_decreasing(self) -> None:
        # Anchored IS always starts at bar 0, so IS length grows per window.
        result = _make_orchestrator(window_type="anchored", n_windows=4).run(
            _SeqFeed(_noisy_uptrend(200))
        )
        is_counts = [w.is_bars for w in result.windows]
        assert is_counts == sorted(is_counts)
        assert is_counts[0] < is_counts[-1]

    def test_rolling_is_bars_roughly_constant(self) -> None:
        result = _make_orchestrator(window_type="rolling", n_windows=4).run(
            _SeqFeed(_noisy_uptrend(200))
        )
        # All non-final rolling windows share the same IS length.
        is_counts = [w.is_bars for w in result.windows[:-1]]
        assert len(set(is_counts)) == 1


# ---------------------------------------------------------------------------
# Stitched OOS curve & lengths
# ---------------------------------------------------------------------------


class TestStitchedCurve:
    @pytest.mark.parametrize("mode", ["rolling", "anchored"])
    def test_oos_length_equals_sum_of_oos_bars(self, mode: str) -> None:
        result = _make_orchestrator(window_type=mode, n_windows=4).run(
            _SeqFeed(_noisy_uptrend(200))
        )
        total_oos = sum(w.oos_bars for w in result.windows)
        assert len(result.oos_equity_curve) == total_oos

    def test_stitched_index_monotonic_increasing(self) -> None:
        result = _make_orchestrator(n_windows=4).run(
            _SeqFeed(_noisy_uptrend(200))
        )
        idx = result.oos_equity_curve.index
        assert idx.is_monotonic_increasing

    def test_stitched_index_is_unique(self) -> None:
        result = _make_orchestrator(n_windows=4).run(
            _SeqFeed(_noisy_uptrend(200))
        )
        assert result.oos_equity_curve.index.is_unique

    def test_first_window_starts_near_initial_cash(self) -> None:
        result = _make_orchestrator(n_windows=4).run(
            _SeqFeed(_noisy_uptrend(200))
        )
        # First stitched point equals the first OOS curve's first equity,
        # which is the initial cash (no fill on the very first OOS bar).
        assert result.oos_equity_curve.iloc[0] == pytest.approx(100_000.0)

    def test_oos_metrics_present(self) -> None:
        result = _make_orchestrator(n_windows=4).run(
            _SeqFeed(_noisy_uptrend(200))
        )
        for key in ("total_return", "sharpe_ratio", "max_drawdown", "num_trades"):
            assert key in result.oos_metrics


# ---------------------------------------------------------------------------
# Decay / robustness
# ---------------------------------------------------------------------------


class TestDecayRobustness:
    @pytest.mark.parametrize("mode", ["rolling", "anchored"])
    def test_decay_scores_length(self, mode: str) -> None:
        result = _make_orchestrator(window_type=mode, n_windows=4).run(
            _SeqFeed(_noisy_uptrend(200))
        )
        assert len(result.decay_scores) == 4
        assert len(result.is_sharpes) == 4
        assert len(result.oos_sharpes) == 4

    def test_decay_equals_is_minus_oos(self) -> None:
        result = _make_orchestrator(n_windows=4).run(
            _SeqFeed(_noisy_uptrend(200))
        )
        for w in result.windows:
            assert w.decay == pytest.approx(w.is_sharpe - w.oos_sharpe)

    def test_robustness_score_is_finite_non_negative(self) -> None:
        result = _make_orchestrator(n_windows=4).run(
            _SeqFeed(_noisy_uptrend(200))
        )
        assert isinstance(result.robustness_score, float)
        assert math.isfinite(result.robustness_score)
        assert result.robustness_score >= 0.0

    def test_robustness_matches_mean_ratio(self) -> None:
        result = _make_orchestrator(n_windows=4).run(
            _SeqFeed(_noisy_uptrend(200))
        )
        mean_is = sum(result.is_sharpes) / len(result.is_sharpes)
        mean_oos = sum(result.oos_sharpes) / len(result.oos_sharpes)
        expected = 0.0 if mean_is == 0 else mean_oos / mean_is
        assert result.robustness_score == pytest.approx(expected)


# ---------------------------------------------------------------------------
# ListFeed behaviour (exercised indirectly via WFO)
# ---------------------------------------------------------------------------


class TestListFeedIndirect:
    def test_multi_trial_reiterates_same_window_feed(self) -> None:
        """
        Each IS window feed drives multiple GridSearch trials (3 combos), and
        the OOS feed drives one backtest.  Correct per-bar equity lengths prove
        the internal ListFeed iterates fresh each run (i.e. resets cleanly).
        """
        result = _make_orchestrator(n_windows=4).run(
            _SeqFeed(_noisy_uptrend(200))
        )
        for w in result.windows:
            # One equity point per OOS bar → iteration produced every bar.
            assert len(w.oos_result.equity_curve) == w.oos_bars
            # The optimizer actually selected params (IS feed was iterable).
            assert "lookback" in w.best_params

    def test_running_twice_on_same_feed_is_stable(self) -> None:
        """Re-running on the same source feed yields identical OOS lengths."""
        feed = _SeqFeed(_noisy_uptrend(200))
        r1 = _make_orchestrator(n_windows=4).run(feed)
        r2 = _make_orchestrator(n_windows=4).run(feed)
        assert len(r1.oos_equity_curve) == len(r2.oos_equity_curve)
