#!/usr/bin/env python3
"""
Phase 7 smoke test — WFOOrchestrator + GridSearch + DictReporter.

Run from the project root:
    python scripts/smoke_test_wfo.py

No pytest.  No source-file modifications.
Prints "ALL CHECKS PASSED" if every assertion is green.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone

# Ensure the project root (parent of this file's directory) is on sys.path
# so that `backtester` is importable when the script is run as
#     python scripts/smoke_test_wfo.py
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from typing import Any, Iterator

import pandas as pd

from backtester.analytics.dict_reporter import DictReporter
from backtester.core.backtester import Backtester
from backtester.execution.commission_models import ZeroCommission
from backtester.execution.execution_models import NextOpenExecution
from backtester.execution.slippage_models import ZeroSlippage
from backtester.interfaces import Bar, BacktestResult, DataFeed, Reporter, Strategy
from backtester.optimize.grid_search import GridSearch
from backtester.optimize.param_space import IntParam, ParamSpace
from backtester.optimize.wfo import ListFeed, WFOOrchestrator, WFOResult
from backtester.risk.risk_models import NoRisk
from backtester.strategy.event_driven import EventDrivenStrategy

_UTC = timezone.utc

# ---------------------------------------------------------------------------
# Minimal test doubles
# ---------------------------------------------------------------------------


class _NoopReporter(Reporter):
    """Reporter that discards results (keeps runs fast and output clean)."""

    def report(self, result: BacktestResult) -> None:  # type: ignore[override]
        return None


class _SeqFeed(DataFeed):
    """Re-iterable list-backed DataFeed used as the outer source feed."""

    def __init__(self, bars: list[Bar]) -> None:
        self._bars: list[Bar] = list(bars)

    def __iter__(self) -> Iterator[Bar]:
        return iter(self._bars)

    def reset(self) -> None:
        pass

    @property
    def symbols(self) -> list[str]:
        return sorted({b.symbol for b in self._bars})


# ---------------------------------------------------------------------------
# Strategy under optimisation
# ---------------------------------------------------------------------------


class SmokeStrategy(EventDrivenStrategy):
    """
    Buys 10 shares on bar ``lookback`` (1-based count within this window),
    sells them 5 bars later if still held.

    ``lookback`` is the hyperparameter varied by the optimizer (IntParam 2–5).
    """

    def __init__(self, lookback: int = 2) -> None:
        super().__init__(lookback=lookback)
        self.lookback: int = lookback
        self._seen: int = 0
        self._bought: bool = False

    def on_bar(self, bar: Bar, context: Any = None) -> None:
        self._seen += 1
        if not self._bought and self._seen >= self.lookback:
            self.buy(bar.symbol, quantity=10)
            self._bought = True
        elif self._bought and self._seen >= self.lookback + 5:
            self.sell(bar.symbol, quantity=10)
            self._bought = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bars(n: int = 60, symbol: str = "TEST") -> list[Bar]:
    """60 bars with linearly increasing close prices (strictly monotone series)."""
    base = datetime(2024, 1, 1, tzinfo=_UTC)
    bars: list[Bar] = []
    for i in range(n):
        price = 100.0 + float(i)
        bars.append(
            Bar(symbol, base + timedelta(days=i),
                price, price + 0.5, price - 0.5, price, 1_000.0)
        )
    return bars


def _total_return(result: BacktestResult) -> float:
    """Objective: maximise total return."""
    return float(result.metrics["total_return"])


def _make_factory():
    """
    Return a backtester_factory as required by WFOOrchestrator.

    The factory is called once with a probe strategy; WFO extracts the
    shared components (execution, slippage, commission, risk, reporter,
    initial_cash) and replaces the feed per window with a ListFeed.
    """
    def factory(strategy: Strategy) -> Backtester:
        slip = ZeroSlippage()
        comm = ZeroCommission()
        return Backtester(
            feed=ListFeed([]),          # placeholder; WFO supplies real feed
            strategy=strategy,
            execution=NextOpenExecution(slip, comm),
            slippage=slip,
            commission=comm,
            risk=NoRisk(),
            reporter=_NoopReporter(),
            initial_cash=100_000.0,
        )
    return factory


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------

_failures: list[str] = []


def check(condition: bool, label: str) -> None:
    """Record a pass/fail for *label*; print immediately."""
    if condition:
        print(f"  ✓  {label}")
    else:
        print(f"  ✗  FAIL — {label}")
        _failures.append(label)


# ---------------------------------------------------------------------------
# 1. GridSearch — exhaustive over IntParam("lookback", 2, 5)
# ---------------------------------------------------------------------------

print("\n── 1. GridSearch (exhaustive) ───────────────────────────────────────")

_bars = _make_bars(60)
_space = ParamSpace([IntParam("lookback", 2, 5)])

_slip = ZeroSlippage()
_comm = ZeroCommission()
_bt_kwargs: dict[str, Any] = {
    "feed": _SeqFeed(_bars),
    "execution": NextOpenExecution(_slip, _comm),
    "slippage": _slip,
    "commission": _comm,
    "risk": NoRisk(),
    "reporter": _NoopReporter(),
    "initial_cash": 100_000.0,
}

_best_params, _best_result = GridSearch().optimize(
    SmokeStrategy, _space, _total_return,
    backtester_kwargs=_bt_kwargs,
)

check(isinstance(_best_result, BacktestResult),
      "optimize() returns a BacktestResult")
check(len(_best_result.equity_curve) > 0,
      "best_result.equity_curve is non-empty")
check(isinstance(_best_params, dict) and "lookback" in _best_params,
      "'lookback' key present in best_params")
check(2 <= _best_params["lookback"] <= 5,
      f"best_params['lookback']={_best_params.get('lookback')} is in [2, 5]")


# ---------------------------------------------------------------------------
# 2. WFOOrchestrator — rolling mode, n_windows=3
# ---------------------------------------------------------------------------

print("\n── 2. WFOOrchestrator (rolling, n_windows=3) ────────────────────────")

_rolling = WFOOrchestrator(
    strategy_class=SmokeStrategy,
    param_space=_space,
    optimizer=GridSearch(),
    objective=_total_return,
    n_trials=10,
    window_type="rolling",
    n_windows=3,
    is_pct=0.6,
    backtester_factory=_make_factory(),
).run(_SeqFeed(_bars))

check(isinstance(_rolling, WFOResult),
      "run() returns a WFOResult")
check(len(_rolling.windows) == 3,
      f"windows count == 3 (got {len(_rolling.windows)})")
check(isinstance(_rolling.oos_equity_curve, pd.Series),
      "oos_equity_curve is a pd.Series")
check(_rolling.oos_equity_curve.index.is_monotonic_increasing,
      "oos_equity_curve index is monotonically increasing")
check(_rolling.oos_equity_curve.index.is_unique,
      "oos_equity_curve index is unique")
check(_rolling.robustness_score >= 0.0,
      f"robustness_score >= 0 (got {_rolling.robustness_score:.4f})")
check("sharpe_ratio" in _rolling.oos_metrics,
      "'sharpe_ratio' present in oos_metrics")
check("total_return" in _rolling.oos_metrics,
      "'total_return' present in oos_metrics")
check(len(_rolling.decay_scores) == 3,
      "decay_scores length == n_windows")
check(len(_rolling.is_sharpes) == 3,
      "is_sharpes length == n_windows")
check(len(_rolling.oos_sharpes) == 3,
      "oos_sharpes length == n_windows")


# ---------------------------------------------------------------------------
# 3. WFOOrchestrator — anchored mode, n_windows=3
# ---------------------------------------------------------------------------

print("\n── 3. WFOOrchestrator (anchored, n_windows=3) ───────────────────────")

_anchored = WFOOrchestrator(
    strategy_class=SmokeStrategy,
    param_space=_space,
    optimizer=GridSearch(),
    objective=_total_return,
    n_trials=10,
    window_type="anchored",
    n_windows=3,
    is_pct=0.6,
    backtester_factory=_make_factory(),
).run(_SeqFeed(_bars))

check(isinstance(_anchored, WFOResult),
      "run() returns a WFOResult")
check(len(_anchored.windows) == 3,
      f"windows count == 3 (got {len(_anchored.windows)})")
check(isinstance(_anchored.oos_equity_curve, pd.Series),
      "oos_equity_curve is a pd.Series")
check(_anchored.oos_equity_curve.index.is_monotonic_increasing,
      "oos_equity_curve index is monotonically increasing")
check(_anchored.oos_equity_curve.index.is_unique,
      "oos_equity_curve index is unique")
check(_anchored.robustness_score >= 0.0,
      f"robustness_score >= 0 (got {_anchored.robustness_score:.4f})")
check("sharpe_ratio" in _anchored.oos_metrics,
      "'sharpe_ratio' present in oos_metrics")
check("total_return" in _anchored.oos_metrics,
      "'total_return' present in oos_metrics")

# Anchored IS bars should be non-decreasing (grows with each window)
_is_counts = [w.is_bars for w in _anchored.windows]
check(_is_counts == sorted(_is_counts) and _is_counts[0] < _is_counts[-1],
      f"anchored IS bars non-decreasing and strictly growing: {_is_counts}")


# ---------------------------------------------------------------------------
# 4. DictReporter — JSON serialisability of best_result
# ---------------------------------------------------------------------------

print("\n── 4. DictReporter — JSON serialisability ───────────────────────────")

_report = DictReporter().report(_best_result)
try:
    json.dumps(_report)
    check(True, "DictReporter output is json.dumps()-able without error")
except (TypeError, ValueError) as exc:
    check(False, f"DictReporter output is json.dumps()-able (raised: {exc})")

check(isinstance(_report, dict), "report() returns a dict")
for _key in ("run_id", "strategy_name", "params", "metrics", "equity_curve"):
    check(_key in _report, f"report dict contains '{_key}'")


# ---------------------------------------------------------------------------
# Final verdict
# ---------------------------------------------------------------------------

print()
if _failures:
    print(f"FAILED  ({len(_failures)} check(s) failed):")
    for _f in _failures:
        print(f"  • {_f}")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
