"""
Tests for backtester/optimize/optuna_optimizer.py.

Covers:
- Returns best params and a valid BacktestResult
- self.study is accessible after optimize()
- n_trials controls the number of Optuna trials
- ConfigurationError on a bad direction value
- ConfigurationError on missing backtester_kwargs
- Optuna's own logging is suppressed (no INFO output during optimize)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator

import optuna
import pytest

from backtester.exceptions import ConfigurationError
from backtester.execution.commission_models import ZeroCommission
from backtester.execution.execution_models import NextOpenExecution
from backtester.execution.slippage_models import ZeroSlippage
from backtester.interfaces import Bar, BacktestResult, DataFeed, Reporter
from backtester.optimize.optuna_optimizer import OptunaOptimizer
from backtester.optimize.param_space import IntParam, ParamSpace
from backtester.risk.risk_models import NoRisk
from backtester.strategy.event_driven import EventDrivenStrategy

_UTC = timezone.utc


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _SeqFeed(DataFeed):
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
    """Buys 10 shares on the ``lookback``-th bar, then holds."""

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


def _uptrend_bars(n: int = 20, symbol: str = "AAPL") -> list[Bar]:
    base = datetime(2024, 1, 1, tzinfo=_UTC)
    bars: list[Bar] = []
    for i in range(n):
        price = 100.0 + i
        bars.append(
            Bar(symbol, base + timedelta(days=i), price, price + 0.5,
                price - 0.5, price, 1_000.0)
        )
    return bars


def _bt_kwargs(bars: list[Bar]) -> dict[str, Any]:
    slip = ZeroSlippage()
    comm = ZeroCommission()
    return {
        "feed": _SeqFeed(bars),
        "execution": NextOpenExecution(slip, comm),
        "slippage": slip,
        "commission": comm,
        "risk": NoRisk(),
        "reporter": _NoopReporter(),
        "initial_cash": 100_000.0,
    }


def _total_return(result: BacktestResult) -> float:
    return float(result.metrics["total_return"])


# ---------------------------------------------------------------------------
# Construction / configuration
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_direction_is_maximize(self) -> None:
        assert OptunaOptimizer().direction == "maximize"

    def test_minimize_allowed(self) -> None:
        assert OptunaOptimizer(direction="minimize").direction == "minimize"

    def test_bad_direction_raises(self) -> None:
        with pytest.raises(ConfigurationError):
            OptunaOptimizer(direction="sideways")

    def test_study_none_before_optimize(self) -> None:
        assert OptunaOptimizer().study is None

    def test_best_result_none_before_optimize(self) -> None:
        assert OptunaOptimizer().best_result is None


# ---------------------------------------------------------------------------
# optimize()
# ---------------------------------------------------------------------------


class TestOptimize:
    def test_returns_params_and_valid_result(self) -> None:
        bars = _uptrend_bars(20)
        space = ParamSpace([IntParam("lookback", 1, 5)])
        opt = OptunaOptimizer(study_name="t1")
        best_params, best_result = opt.optimize(
            EntryDelayStrategy, space, _total_return,
            n_trials=8, backtester_kwargs=_bt_kwargs(bars),
        )
        assert isinstance(best_params, dict)
        assert "lookback" in best_params
        assert isinstance(best_result, BacktestResult)

    def test_study_accessible_after_optimize(self) -> None:
        bars = _uptrend_bars(20)
        space = ParamSpace([IntParam("lookback", 1, 5)])
        opt = OptunaOptimizer()
        opt.optimize(
            EntryDelayStrategy, space, _total_return,
            n_trials=5, backtester_kwargs=_bt_kwargs(bars),
        )
        assert isinstance(opt.study, optuna.Study)

    def test_best_result_stored_on_instance(self) -> None:
        bars = _uptrend_bars(20)
        space = ParamSpace([IntParam("lookback", 1, 5)])
        opt = OptunaOptimizer()
        _, best_result = opt.optimize(
            EntryDelayStrategy, space, _total_return,
            n_trials=6, backtester_kwargs=_bt_kwargs(bars),
        )
        assert opt.best_result is best_result

    def test_n_trials_controls_trial_count(self) -> None:
        bars = _uptrend_bars(20)
        space = ParamSpace([IntParam("lookback", 1, 20)])
        opt = OptunaOptimizer()
        opt.optimize(
            EntryDelayStrategy, space, _total_return,
            n_trials=7, backtester_kwargs=_bt_kwargs(bars),
        )
        assert len(opt.study.trials) == 7

    def test_best_params_validate_against_space(self) -> None:
        bars = _uptrend_bars(20)
        space = ParamSpace([IntParam("lookback", 1, 5)])
        opt = OptunaOptimizer()
        best_params, _ = opt.optimize(
            EntryDelayStrategy, space, _total_return,
            n_trials=8, backtester_kwargs=_bt_kwargs(bars),
        )
        space.validate(best_params)  # must not raise

    def test_missing_backtester_kwargs_raises(self) -> None:
        space = ParamSpace([IntParam("lookback", 1, 5)])
        with pytest.raises(ConfigurationError):
            OptunaOptimizer().optimize(
                EntryDelayStrategy, space, _total_return, n_trials=3,
            )


# ---------------------------------------------------------------------------
# Logging suppression
# ---------------------------------------------------------------------------


class TestLoggingSuppression:
    def test_no_info_logs_during_optimize(self) -> None:
        bars = _uptrend_bars(20)
        space = ParamSpace([IntParam("lookback", 1, 5)])

        records: list[logging.LogRecord] = []

        class _Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        optuna_logger = logging.getLogger("optuna")
        handler = _Capture()
        optuna_logger.addHandler(handler)
        try:
            OptunaOptimizer().optimize(
                EntryDelayStrategy, space, _total_return,
                n_trials=5, backtester_kwargs=_bt_kwargs(bars),
            )
        finally:
            optuna_logger.removeHandler(handler)

        info_records = [r for r in records if r.levelno <= logging.INFO]
        assert info_records == []

    def test_verbosity_is_warning_after_optimize(self) -> None:
        bars = _uptrend_bars(20)
        space = ParamSpace([IntParam("lookback", 1, 5)])
        OptunaOptimizer().optimize(
            EntryDelayStrategy, space, _total_return,
            n_trials=3, backtester_kwargs=_bt_kwargs(bars),
        )
        assert optuna.logging.get_verbosity() == optuna.logging.WARNING
