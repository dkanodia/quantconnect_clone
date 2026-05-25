"""
Tests for backtester/optimize/grid_search.py and the shared ParamSpace.

Covers:
- Exhaustive search finds the best params on a trivial monotonic objective
- The returned best result has the highest objective of every combination
- n_trials sampling evaluates exactly n_trials combinations
- ConfigurationError on missing / incomplete backtester_kwargs
- ParamSpace.grid_combinations / validate behaviour and bound validation
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Iterator, Optional

import pytest

from backtester.exceptions import ConfigurationError
from backtester.execution.commission_models import ZeroCommission
from backtester.execution.execution_models import NextOpenExecution
from backtester.execution.slippage_models import ZeroSlippage
from backtester.interfaces import Bar, BacktestResult, DataFeed, Reporter
from backtester.optimize.grid_search import GridSearch
from backtester.optimize.param_space import (
    CategoricalParam,
    FloatParam,
    IntParam,
    ParamSpace,
)
from backtester.risk.risk_models import NoRisk
from backtester.strategy.event_driven import EventDrivenStrategy

_UTC = timezone.utc


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _SeqFeed(DataFeed):
    """Minimal re-iterable DataFeed wrapping a list of bars (test-local)."""

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
    """Reporter that does nothing (keeps optimisation fast)."""

    def report(self, result: BacktestResult) -> None:
        return None


class EntryDelayStrategy(EventDrivenStrategy):
    """
    Buys a fixed quantity once, on the ``lookback``-th bar, then holds.

    On a strictly rising series an earlier entry (smaller ``lookback``)
    captures more upside, so ``total_return`` is monotonically decreasing in
    ``lookback`` — the optimal value is the smallest ``lookback``.
    """

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


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _uptrend_bars(n: int = 20, symbol: str = "AAPL") -> list[Bar]:
    """Strictly rising open=close series → monotonic objective in lookback."""
    base = datetime(2024, 1, 1, tzinfo=_UTC)
    bars: list[Bar] = []
    for i in range(n):
        price = 100.0 + i  # strictly increasing
        bars.append(
            Bar(
                symbol=symbol,
                timestamp=base + timedelta(days=i),
                open=price,
                high=price + 0.5,
                low=price - 0.5,
                close=price,
                volume=1_000.0,
            )
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
# ParamSpace
# ---------------------------------------------------------------------------


class TestParamSpaceGrid:
    def test_int_param_inclusive_range(self) -> None:
        ps = ParamSpace([IntParam("k", 1, 3)])
        combos = ps.grid_combinations()
        assert [c["k"] for c in combos] == [1, 2, 3]

    def test_int_param_step(self) -> None:
        ps = ParamSpace([IntParam("k", 0, 10, step=5)])
        assert [c["k"] for c in ps.grid_combinations()] == [0, 5, 10]

    def test_float_param_has_ten_points(self) -> None:
        ps = ParamSpace([FloatParam("x", 0.0, 1.0)])
        combos = ps.grid_combinations()
        assert len(combos) == 10
        assert combos[0]["x"] == pytest.approx(0.0)
        assert combos[-1]["x"] == pytest.approx(1.0)

    def test_float_param_values_are_python_floats(self) -> None:
        ps = ParamSpace([FloatParam("x", 0.0, 1.0)])
        for c in ps.grid_combinations():
            assert type(c["x"]) is float

    def test_categorical_param(self) -> None:
        ps = ParamSpace([CategoricalParam("m", ["a", "b", "c"])])
        assert [c["m"] for c in ps.grid_combinations()] == ["a", "b", "c"]

    def test_cartesian_product_size(self) -> None:
        ps = ParamSpace([
            IntParam("k", 1, 3),          # 3
            CategoricalParam("m", ["a", "b"]),  # 2
        ])
        assert len(ps.grid_combinations()) == 6

    def test_empty_space_raises(self) -> None:
        with pytest.raises(ConfigurationError):
            ParamSpace([]).grid_combinations()

    def test_int_bad_bounds_raises(self) -> None:
        with pytest.raises(ConfigurationError):
            ParamSpace([IntParam("k", 5, 1)]).grid_combinations()

    def test_int_bad_step_raises(self) -> None:
        with pytest.raises(ConfigurationError):
            ParamSpace([IntParam("k", 1, 5, step=0)]).grid_combinations()

    def test_float_bad_bounds_raises(self) -> None:
        with pytest.raises(ConfigurationError):
            ParamSpace([FloatParam("x", 1.0, 0.0)]).grid_combinations()

    def test_float_log_requires_positive_low(self) -> None:
        with pytest.raises(ConfigurationError):
            ParamSpace([FloatParam("x", 0.0, 1.0, log=True)]).grid_combinations()

    def test_empty_categorical_raises(self) -> None:
        with pytest.raises(ConfigurationError):
            ParamSpace([CategoricalParam("m", [])]).grid_combinations()


class TestParamSpaceValidate:
    def test_valid_params_pass(self) -> None:
        ps = ParamSpace([IntParam("k", 1, 5), CategoricalParam("m", ["a", "b"])])
        ps.validate({"k": 3, "m": "a"})  # no raise

    def test_missing_param_raises(self) -> None:
        ps = ParamSpace([IntParam("k", 1, 5)])
        with pytest.raises(ConfigurationError):
            ps.validate({})

    def test_int_out_of_bounds_raises(self) -> None:
        ps = ParamSpace([IntParam("k", 1, 5)])
        with pytest.raises(ConfigurationError):
            ps.validate({"k": 9})

    def test_categorical_not_in_choices_raises(self) -> None:
        ps = ParamSpace([CategoricalParam("m", ["a", "b"])])
        with pytest.raises(ConfigurationError):
            ps.validate({"m": "z"})


# ---------------------------------------------------------------------------
# GridSearch
# ---------------------------------------------------------------------------


class TestGridSearchExhaustive:
    def test_finds_best_params_on_monotonic_objective(self) -> None:
        bars = _uptrend_bars(20)
        space = ParamSpace([IntParam("lookback", 1, 5)])
        best_params, best_result = GridSearch().optimize(
            EntryDelayStrategy,
            space,
            objective=_total_return,
            backtester_kwargs=_bt_kwargs(bars),
        )
        # Earlier entry → higher return → best lookback is the smallest.
        assert best_params == {"lookback": 1}
        assert isinstance(best_result, BacktestResult)

    def test_best_result_beats_every_other_combination(self) -> None:
        bars = _uptrend_bars(20)
        space = ParamSpace([IntParam("lookback", 1, 5)])
        _, best_result = GridSearch().optimize(
            EntryDelayStrategy,
            space,
            objective=_total_return,
            backtester_kwargs=_bt_kwargs(bars),
        )
        best_score = _total_return(best_result)

        # Independently evaluate every combination.
        from backtester.core.backtester import Backtester

        for combo in space.grid_combinations():
            strat = EntryDelayStrategy(**combo)
            result = Backtester(strategy=strat, **_bt_kwargs(bars)).run()
            assert best_score >= _total_return(result) - 1e-12

    def test_returns_tuple_of_dict_and_result(self) -> None:
        bars = _uptrend_bars(15)
        space = ParamSpace([IntParam("lookback", 1, 3)])
        out = GridSearch().optimize(
            EntryDelayStrategy, space, _total_return,
            backtester_kwargs=_bt_kwargs(bars),
        )
        assert isinstance(out, tuple) and len(out) == 2
        assert isinstance(out[0], dict)
        assert isinstance(out[1], BacktestResult)


class TestGridSearchSampling:
    def test_n_trials_evaluates_exactly_n_combinations(self) -> None:
        bars = _uptrend_bars(15)
        space = ParamSpace([IntParam("lookback", 1, 10)])  # 10 combos
        calls: list[int] = []

        def counting_objective(result: BacktestResult) -> float:
            calls.append(1)
            return _total_return(result)

        GridSearch().optimize(
            EntryDelayStrategy,
            space,
            counting_objective,
            n_trials=4,
            backtester_kwargs=_bt_kwargs(bars),
        )
        assert len(calls) == 4

    def test_n_trials_larger_than_grid_runs_full_grid(self) -> None:
        bars = _uptrend_bars(15)
        space = ParamSpace([IntParam("lookback", 1, 3)])  # 3 combos
        calls: list[int] = []

        def counting_objective(result: BacktestResult) -> float:
            calls.append(1)
            return _total_return(result)

        GridSearch().optimize(
            EntryDelayStrategy,
            space,
            counting_objective,
            n_trials=99,
            backtester_kwargs=_bt_kwargs(bars),
        )
        assert len(calls) == 3


class TestGridSearchConfigErrors:
    def test_missing_backtester_kwargs_raises(self) -> None:
        space = ParamSpace([IntParam("lookback", 1, 3)])
        with pytest.raises(ConfigurationError):
            GridSearch().optimize(EntryDelayStrategy, space, _total_return)

    def test_none_backtester_kwargs_raises(self) -> None:
        space = ParamSpace([IntParam("lookback", 1, 3)])
        with pytest.raises(ConfigurationError):
            GridSearch().optimize(
                EntryDelayStrategy, space, _total_return,
                backtester_kwargs=None,
            )

    def test_incomplete_backtester_kwargs_raises(self) -> None:
        bars = _uptrend_bars(10)
        kwargs = _bt_kwargs(bars)
        del kwargs["risk"]  # drop a required component
        space = ParamSpace([IntParam("lookback", 1, 3)])
        with pytest.raises(ConfigurationError):
            GridSearch().optimize(
                EntryDelayStrategy, space, _total_return,
                backtester_kwargs=kwargs,
            )
