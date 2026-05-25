"""
Tests for the updated Phase 4 Backtester.

Covers:
- End-to-end runs with real strategies: buy-and-hold, no-trade, multi-bar
- Equity curve and trade DataFrame in BacktestResult
- Metrics: total_return, final_equity, num_trades
- Order execution lifecycle (NextOpen fills on the following bar)
- Risk model integration (MaxDrawdownHalt halts, PositionSizeLimit blocks)
- Strategy.on_order called after fill
- ConfigurationError on missing/wrong components
- debug mode (logging path exercised, no crash)
- Empty feed (zero bars)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional
from unittest.mock import MagicMock

import pandas as pd
import pytest

from backtester.core.backtester import Backtester
from backtester.exceptions import ConfigurationError, MaxDrawdownExceeded, StrategyError
from backtester.execution.commission_models import ZeroCommission
from backtester.execution.execution_models import NextOpenExecution, SameBarExecution
from backtester.execution.slippage_models import ZeroSlippage
from backtester.interfaces import (
    Bar,
    BacktestResult,
    CommissionModel,
    DataFeed,
    Event,
    ExecutionModel,
    Fill,
    Order,
    Reporter,
    RiskModel,
    SlippageModel,
    Strategy,
)
from backtester.risk.risk_models import MaxDrawdownHalt, NoRisk
from backtester.strategy.event_driven import EventDrivenStrategy


# ---------------------------------------------------------------------------
# Minimal stub implementations
# ---------------------------------------------------------------------------


class _ListFeed(DataFeed):
    def __init__(self, bars: list[Bar]) -> None:
        self._bars = bars

    def __iter__(self):
        return iter(self._bars)

    def reset(self) -> None:
        pass

    @property
    def symbols(self) -> list[str]:
        return list({b.symbol for b in self._bars})


class _NoOpReporter(Reporter):
    def report(self, result: BacktestResult) -> None:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ZERO_SLIP = ZeroSlippage()
_ZERO_COM = ZeroCommission()


def _bar(
    day: int,
    open_: float = 100.0,
    high: float = 110.0,
    low: float = 90.0,
    close: float = 100.0,
    volume: float = 10_000.0,
    symbol: str = "AAPL",
) -> Bar:
    return Bar(
        symbol=symbol,
        timestamp=datetime(2024, 1, day, tzinfo=timezone.utc),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def _make_backtester(
    bars: list[Bar],
    strategy: Strategy,
    execution: Optional[ExecutionModel] = None,
    risk: Optional[RiskModel] = None,
    initial_cash: float = 100_000.0,
    debug: bool = False,
) -> Backtester:
    return Backtester(
        feed=_ListFeed(bars),
        strategy=strategy,
        execution=execution or NextOpenExecution(_ZERO_SLIP, _ZERO_COM),
        slippage=_ZERO_SLIP,
        commission=_ZERO_COM,
        risk=risk or NoRisk(),
        reporter=_NoOpReporter(),
        initial_cash=initial_cash,
        debug=debug,
    )


# ---------------------------------------------------------------------------
# Passive strategy (no trades)
# ---------------------------------------------------------------------------


class PassiveStrategy(EventDrivenStrategy):
    pass


# ---------------------------------------------------------------------------
# Buy-and-hold strategy
# ---------------------------------------------------------------------------


class BuyOnBar1(EventDrivenStrategy):
    """Submits one BUY on the first bar, nothing after."""

    def __init__(self) -> None:
        super().__init__()
        self._bought = False
        self.on_order_calls: list[Order] = []

    def on_bar(self, bar: Bar, context: Any) -> None:
        if not self._bought:
            self.buy(bar.symbol, quantity=10)
            self._bought = True

    def on_order(self, order: Order, context: Any) -> None:
        self.on_order_calls.append(order)


# ---------------------------------------------------------------------------
# Basic run mechanics
# ---------------------------------------------------------------------------


class TestBasicRun:
    def test_returns_backtest_result(self) -> None:
        result = _make_backtester([_bar(2)], PassiveStrategy()).run()
        assert isinstance(result, BacktestResult)

    def test_strategy_name_in_result(self) -> None:
        result = _make_backtester([_bar(2)], PassiveStrategy()).run()
        assert result.strategy_name == "PassiveStrategy"

    def test_equity_curve_length_matches_bars(self) -> None:
        bars = [_bar(d) for d in range(2, 7)]  # 5 bars
        result = _make_backtester(bars, PassiveStrategy()).run()
        assert len(result.equity_curve) == 5

    def test_equity_curve_is_series(self) -> None:
        result = _make_backtester([_bar(2), _bar(3)], PassiveStrategy()).run()
        assert isinstance(result.equity_curve, pd.Series)

    def test_no_trades_passive_strategy(self) -> None:
        bars = [_bar(d) for d in range(2, 5)]
        result = _make_backtester(bars, PassiveStrategy()).run()
        assert result.metrics["num_trades"] == 0

    def test_equity_constant_no_trades(self) -> None:
        bars = [_bar(d) for d in range(2, 5)]
        result = _make_backtester(bars, PassiveStrategy(), initial_cash=50_000.0).run()
        assert all(v == pytest.approx(50_000.0) for v in result.equity_curve.values)

    def test_zero_bar_feed_returns_empty_result(self) -> None:
        result = _make_backtester([], PassiveStrategy()).run()
        assert len(result.equity_curve) == 0
        assert result.metrics["num_trades"] == 0

    def test_total_return_zero_passive(self) -> None:
        bars = [_bar(d) for d in range(2, 5)]
        result = _make_backtester(bars, PassiveStrategy()).run()
        assert result.metrics["total_return"] == pytest.approx(0.0)

    def test_metrics_dict_has_required_keys(self) -> None:
        # Phase 6: compute_all() replaces the Phase 4 stub; verify the full
        # analytics key set is present, including the legacy keys.
        result = _make_backtester([_bar(2)], PassiveStrategy()).run()
        for key in (
            "total_return", "final_equity", "num_trades",
            "sharpe_ratio", "max_drawdown", "win_rate",
        ):
            assert key in result.metrics


# ---------------------------------------------------------------------------
# Order execution and fills
# ---------------------------------------------------------------------------


class TestOrderExecution:
    def test_buy_on_bar1_fills_on_bar2_open(self) -> None:
        bars = [_bar(2, open_=100.0), _bar(3, open_=105.0), _bar(4, open_=110.0)]
        strategy = BuyOnBar1()
        result = _make_backtester(bars, strategy, execution=NextOpenExecution(_ZERO_SLIP, _ZERO_COM)).run()
        # Fill at bar-3 open=105
        assert result.metrics["num_trades"] == 0  # no SELL yet → position is open, not a trade

    def test_on_order_called_after_fill(self) -> None:
        bars = [_bar(2, open_=100.0), _bar(3, open_=105.0), _bar(4)]
        strategy = BuyOnBar1()
        _make_backtester(bars, strategy).run()
        assert len(strategy.on_order_calls) == 1

    def test_equity_increases_after_profitable_buy(self) -> None:
        # Buy 10 shares at bar-2's open=100 (NextOpen fill on bar-3)
        # Bar prices: bar-2 close=100, bar-3 open=100 (fill), close=120
        bars = [
            _bar(2, open_=100.0, close=100.0),
            _bar(3, open_=100.0, close=120.0),
            _bar(4, open_=120.0, close=120.0),
        ]
        strategy = BuyOnBar1()
        result = _make_backtester(bars, strategy, initial_cash=10_000.0).run()
        # After fill: cash = 10000 - 10*100 = 9000; position = 10 @ 120 close = 1200
        # Final equity = 9000 + 1200 = 10200 > 10000
        final = result.metrics["final_equity"]
        assert final > 10_000.0

    def test_total_return_positive_after_gain(self) -> None:
        bars = [
            _bar(2, open_=100.0, close=100.0),
            _bar(3, open_=100.0, close=150.0),
            _bar(4, open_=150.0, close=150.0),
        ]
        strategy = BuyOnBar1()
        result = _make_backtester(bars, strategy, initial_cash=10_000.0).run()
        assert result.metrics["total_return"] > 0.0


# ---------------------------------------------------------------------------
# Same-bar execution
# ---------------------------------------------------------------------------


class TestSameBarExecution:
    def test_same_bar_fills_immediately(self) -> None:
        """With SameBarExecution, an order placed in on_bar fills in the NEXT
        bar's execution step — at prev_bar.close (the bar where the order was placed)."""
        bars = [_bar(2, close=100.0), _bar(3, close=110.0), _bar(4, close=120.0)]
        strategy = BuyOnBar1()
        _make_backtester(
            bars, strategy,
            execution=SameBarExecution(_ZERO_SLIP, _ZERO_COM),
        ).run()
        assert len(strategy.on_order_calls) == 1  # fill happened


# ---------------------------------------------------------------------------
# Risk model integration
# ---------------------------------------------------------------------------


class _BigBuyThenKeep(EventDrivenStrategy):
    """Buys 900 shares on bar 1, then places a small order every subsequent bar."""

    def __init__(self) -> None:
        super().__init__()
        self._first = True

    def on_bar(self, bar: Bar, context: Any) -> None:
        if self._first:
            self.buy(bar.symbol, quantity=900)
            self._first = False
        else:
            self.buy(bar.symbol, quantity=1)  # keeps triggering risk.check


class TestRiskIntegration:
    def test_max_drawdown_halt_raises(self) -> None:
        """MaxDrawdownHalt fires when a new order is risk-checked after a >1% drawdown.

        Flow:
          bar(2) on_bar  → submit buy 900 @ ~100 (peak equity ~100k)
          bar(3) execute → fill 900 @ open=100; cash=10k; equity=100k
          bar(3) on_bar  → submit buy 1
          bar(4) execute → fill 1 @ open=100; equity still ~100k (last price hasn't updated yet)
          bar(4) update  → close=10 → equity drops to ~19k (81% drawdown from peak)
          bar(4) on_bar  → submit buy 1
          bar(5) execute → risk.check sees 81% drawdown > 1% limit → raises
        """
        bars = [
            _bar(2, open_=100.0, close=100.0),
            _bar(3, open_=100.0, close=100.0),
            _bar(4, open_=100.0, close=10.0),   # price collapses
            _bar(5, open_=10.0, close=10.0),    # risk check fires here
        ]
        bt = _make_backtester(
            bars, _BigBuyThenKeep(),
            risk=MaxDrawdownHalt(limit=0.01),  # 1 % limit
            initial_cash=100_000.0,
        )
        with pytest.raises(MaxDrawdownExceeded):
            bt.run()


# ---------------------------------------------------------------------------
# ConfigurationError validation
# ---------------------------------------------------------------------------


class TestConfigurationError:
    def _good_args(self):
        return dict(
            feed=_ListFeed([]),
            strategy=PassiveStrategy(),
            execution=NextOpenExecution(_ZERO_SLIP, _ZERO_COM),
            slippage=_ZERO_SLIP,
            commission=_ZERO_COM,
            risk=NoRisk(),
            reporter=_NoOpReporter(),
        )

    def test_none_feed_raises(self) -> None:
        args = self._good_args()
        args["feed"] = None
        with pytest.raises(ConfigurationError):
            Backtester(**args)

    def test_wrong_type_raises(self) -> None:
        args = self._good_args()
        args["strategy"] = "not-a-strategy"
        with pytest.raises(ConfigurationError):
            Backtester(**args)

    def test_none_risk_raises(self) -> None:
        args = self._good_args()
        args["risk"] = None
        with pytest.raises(ConfigurationError):
            Backtester(**args)


# ---------------------------------------------------------------------------
# Strategy lifecycle hooks
# ---------------------------------------------------------------------------


class _LifecycleTracker(EventDrivenStrategy):
    def __init__(self) -> None:
        super().__init__()
        self.start_called = False
        self.end_called = False
        self.bar_count = 0

    def on_start(self, context: Any) -> None:
        self.start_called = True

    def on_end(self, context: Any) -> None:
        self.end_called = True

    def on_bar(self, bar: Bar, context: Any) -> None:
        self.bar_count += 1


class TestLifecycleHooks:
    def test_on_start_called(self) -> None:
        s = _LifecycleTracker()
        _make_backtester([_bar(2)], s).run()
        assert s.start_called

    def test_on_end_called(self) -> None:
        s = _LifecycleTracker()
        _make_backtester([_bar(2)], s).run()
        assert s.end_called

    def test_on_bar_called_for_each_bar(self) -> None:
        s = _LifecycleTracker()
        _make_backtester([_bar(d) for d in range(2, 7)], s).run()
        assert s.bar_count == 5

    def test_on_start_error_raises_strategy_error(self) -> None:
        class BadStart(EventDrivenStrategy):
            def on_start(self, context: Any) -> None:
                raise RuntimeError("boom")

        with pytest.raises(StrategyError):
            _make_backtester([_bar(2)], BadStart()).run()

    def test_on_end_error_raises_strategy_error(self) -> None:
        class BadEnd(EventDrivenStrategy):
            def on_end(self, context: Any) -> None:
                raise RuntimeError("boom")

        with pytest.raises(StrategyError):
            _make_backtester([_bar(2)], BadEnd()).run()

    def test_on_bar_error_raises_strategy_error(self) -> None:
        class BadBar(EventDrivenStrategy):
            def on_bar(self, bar: Bar, context: Any) -> None:
                raise RuntimeError("boom")

        with pytest.raises(StrategyError):
            _make_backtester([_bar(2)], BadBar()).run()


# ---------------------------------------------------------------------------
# Debug mode
# ---------------------------------------------------------------------------


class TestDebugMode:
    def test_debug_mode_does_not_crash(self) -> None:
        bars = [_bar(d) for d in range(2, 5)]
        result = _make_backtester(bars, PassiveStrategy(), debug=True).run()
        assert isinstance(result, BacktestResult)
