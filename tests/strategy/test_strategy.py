"""
Phase 3 tests — strategy API and Backtester skeleton.

Covers:
- BaseStrategy: params, buy/sell/cancel, flush, context injection,
  current_bar, order validation
- EventDrivenStrategy: warmup, bar_index, get_signals guard
- VectorizedStrategy: on_bar guard, get_signals contract
- Backtester: construction validation, run() lifecycle, debug logging,
  pending-order accumulation
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Iterator, Optional, Sequence
from unittest.mock import patch

import pandas as pd
import pytest

from backtester.core.backtester import Backtester
from backtester.exceptions import (
    BacktestError,
    ConfigurationError,
    InvalidOrderError,
    StrategyError,
)
from backtester.interfaces import (
    Bar,
    BacktestResult,
    CommissionModel,
    DataFeed,
    Event,
    EventType,
    ExecutionModel,
    Fill,
    Order,
    OrderSide,
    OrderType,
    PortfolioTracker,
    Position,
    Reporter,
    RiskModel,
    SlippageModel,
)
from backtester.strategy.base import BaseStrategy
from backtester.strategy.event_driven import EventDrivenStrategy
from backtester.strategy.vectorized import VectorizedStrategy

_UTC = timezone.utc


# ---------------------------------------------------------------------------
# Shared test fixtures — minimal concrete implementations
# ---------------------------------------------------------------------------


def _bar(symbol: str = "SPY", idx: int = 1) -> Bar:
    return Bar(
        symbol=symbol,
        timestamp=datetime(2024, 1, idx, tzinfo=_UTC),
        open=100.0 + idx,
        high=105.0 + idx,
        low=98.0 + idx,
        close=103.0 + idx,
        volume=1_000_000.0,
    )


def _make_bars(n: int = 5, symbol: str = "SPY") -> list[Bar]:
    return [_bar(symbol, i + 1) for i in range(n)]


class _Feed(DataFeed):
    def __init__(self, bars: list[Bar]) -> None:
        self._bars = bars

    def __iter__(self) -> Iterator[Bar]:
        return iter(self._bars)

    def reset(self) -> None:
        pass

    @property
    def symbols(self) -> list[str]:
        return list({b.symbol for b in self._bars})


class _Execution(ExecutionModel):
    def execute(
        self, order: Order, current_bar: Bar, next_bar: Optional[Bar]
    ) -> Optional[Fill]:
        return None


class _Slippage(SlippageModel):
    def adjust(self, raw_price: float, order: Order, bar: Bar) -> float:
        return raw_price


class _Commission(CommissionModel):
    def compute(self, fill: Fill) -> float:
        return 0.0


class _Portfolio(PortfolioTracker):
    def apply_fill(self, fill: Fill) -> None: pass
    def update_market(self, bars: Sequence[Bar]) -> None: pass
    def record_equity(self, timestamp: datetime) -> None: pass
    def get_equity_curve(self) -> pd.Series: return pd.Series(dtype=float)
    def get_positions(self) -> dict[str, Position]: return {}
    def get_cash(self) -> float: return 100_000.0
    def get_total_equity(self) -> float: return 100_000.0
    def get_trades(self) -> pd.DataFrame: return pd.DataFrame()


class _Risk(RiskModel):
    def check(self, order: Order, portfolio: Any, bar: Bar) -> bool:
        return True
    def on_fill(self, fill: Fill, portfolio: Any) -> None:
        pass


class _Reporter(Reporter):
    def report(self, result: BacktestResult) -> Any:
        return result


def _make_backtester(strategy: Any, bars: Optional[list[Bar]] = None, debug: bool = False) -> Backtester:
    """Return a Backtester wired with all-stub components."""
    return Backtester(
        feed=_Feed(_make_bars() if bars is None else bars),
        strategy=strategy,
        execution=_Execution(),
        slippage=_Slippage(),
        commission=_Commission(),
        risk=_Risk(),
        reporter=_Reporter(),
        debug=debug,
    )


# ---------------------------------------------------------------------------
# BaseStrategy
# ---------------------------------------------------------------------------


class TestBaseStrategy:
    def test_params_stored_at_construction(self) -> None:
        s = BaseStrategy(lookback=20, fast=5)
        assert s.params == {"lookback": 20, "fast": 5}

    def test_empty_params(self) -> None:
        s = BaseStrategy()
        assert s.params == {}

    def test_buy_enqueues_market_order(self) -> None:
        s = BaseStrategy()
        oid = s.buy("SPY", 100)
        orders = s._flush_orders()
        assert len(orders) == 1
        assert orders[0].side == OrderSide.BUY
        assert orders[0].quantity == 100
        assert orders[0].type == OrderType.MARKET
        assert orders[0].id == oid

    def test_sell_enqueues_market_order(self) -> None:
        s = BaseStrategy()
        oid = s.sell("SPY", 50)
        orders = s._flush_orders()
        assert len(orders) == 1
        assert orders[0].side == OrderSide.SELL
        assert orders[0].quantity == 50
        assert orders[0].id == oid

    def test_buy_limit_order(self) -> None:
        s = BaseStrategy()
        s.buy("AAPL", 10, order_type=OrderType.LIMIT, limit_price=150.0)
        order = s._flush_orders()[0]
        assert order.type == OrderType.LIMIT
        assert order.limit_price == pytest.approx(150.0)

    def test_sell_stop_order(self) -> None:
        s = BaseStrategy()
        s.sell("TSLA", 5, order_type=OrderType.STOP, stop_price=200.0)
        order = s._flush_orders()[0]
        assert order.type == OrderType.STOP
        assert order.stop_price == pytest.approx(200.0)

    def test_buy_limit_without_price_raises(self) -> None:
        s = BaseStrategy()
        with pytest.raises(InvalidOrderError):
            s.buy("SPY", 10, order_type=OrderType.LIMIT)

    def test_sell_stop_without_price_raises(self) -> None:
        s = BaseStrategy()
        with pytest.raises(InvalidOrderError):
            s.sell("SPY", 10, order_type=OrderType.STOP)

    def test_flush_orders_clears_queue(self) -> None:
        s = BaseStrategy()
        s.buy("SPY", 10)
        s._flush_orders()
        assert s._flush_orders() == []

    def test_cancel_enqueues_cancellation(self) -> None:
        s = BaseStrategy()
        s.cancel("order-123")
        cancels = s._flush_cancellations()
        assert "order-123" in cancels

    def test_flush_cancellations_clears_queue(self) -> None:
        s = BaseStrategy()
        s.cancel("order-abc")
        s._flush_cancellations()
        assert s._flush_cancellations() == []

    def test_current_bar_is_none_before_loop(self) -> None:
        s = BaseStrategy()
        assert s.current_bar is None

    def test_update_current_bar(self) -> None:
        s = BaseStrategy()
        bar = _bar()
        s._update_current_bar(bar)
        assert s.current_bar is bar

    def test_set_context_stores_components(self) -> None:
        s = BaseStrategy()
        feed = _Feed([])
        bus = object()
        s._set_context(feed=feed, portfolio=None, event_bus=bus)
        assert s._feed is feed
        assert s._event_bus is bus
        assert s._portfolio is None

    def test_should_process_bar_always_true(self) -> None:
        s = BaseStrategy()
        for i in range(1, 6):
            assert s._should_process_bar(i) is True

    def test_is_not_vectorized(self) -> None:
        s = BaseStrategy()
        assert s._is_vectorized() is False

    def test_buy_returns_unique_order_ids(self) -> None:
        s = BaseStrategy()
        ids = {s.buy("SPY", 1) for _ in range(10)}
        assert len(ids) == 10  # all unique UUIDs

    def test_multiple_orders_accumulated(self) -> None:
        s = BaseStrategy()
        s.buy("SPY", 10)
        s.sell("SPY", 5)
        s.buy("AAPL", 3)
        orders = s._flush_orders()
        assert len(orders) == 3

    def test_hook_defaults_are_noop(self) -> None:
        s = BaseStrategy()
        bar = _bar()
        order = Order("SPY", OrderSide.BUY, OrderType.MARKET, 1)
        # None of these should raise.
        s.on_start(None)
        s.on_bar(bar, None)
        s.on_order(order, None)
        s.on_end(None)

    def test_get_signals_returns_empty_series(self) -> None:
        s = BaseStrategy()
        result = s.get_signals(pd.DataFrame({"close": [1, 2, 3]}))
        assert isinstance(result, pd.Series)
        assert result.empty


# ---------------------------------------------------------------------------
# EventDrivenStrategy
# ---------------------------------------------------------------------------


class _TrackingStrategy(EventDrivenStrategy):
    """Concrete strategy that records lifecycle calls for assertions."""

    def __init__(self, warmup_period: int = 0, **params: Any) -> None:
        super().__init__(warmup_period=warmup_period, **params)
        self.started = False
        self.ended = False
        self.bars_seen: list[Bar] = []
        self.orders_submitted: list[tuple[str, int]] = []  # (symbol, qty)

    def on_start(self, context: Any = None) -> None:
        self.started = True

    def on_bar(self, bar: Bar, context: Any = None) -> None:
        self.bars_seen.append(bar)
        self.orders_submitted.append((bar.symbol, 1))
        self.buy(bar.symbol, 1)

    def on_end(self, context: Any = None) -> None:
        self.ended = True


class TestEventDrivenStrategy:
    def test_bar_index_increments_every_bar(self) -> None:
        s = EventDrivenStrategy()
        for i in range(1, 6):
            s._should_process_bar(i)
        assert s.bar_index == 5

    def test_no_warmup_processes_all_bars(self) -> None:
        s = EventDrivenStrategy(warmup_period=0)
        results = [s._should_process_bar(i) for i in range(1, 6)]
        assert all(results)

    def test_warmup_skips_initial_bars(self) -> None:
        s = EventDrivenStrategy(warmup_period=3)
        results = [s._should_process_bar(i) for i in range(1, 7)]
        # bars 1,2,3 → skipped; bars 4,5,6 → processed
        assert results == [False, False, False, True, True, True]

    def test_bar_index_updated_even_during_warmup(self) -> None:
        s = EventDrivenStrategy(warmup_period=5)
        for i in range(1, 4):
            s._should_process_bar(i)
        # bar_index tracks ALL bars, not just processed ones
        assert s.bar_index == 3

    def test_warmup_stored_in_params_separately(self) -> None:
        s = EventDrivenStrategy(warmup_period=10, lookback=20)
        assert s.warmup_period == 10
        assert s.params.get("lookback") == 20
        # warmup_period is NOT in params (it's an internal concept)
        assert "warmup_period" not in s.params

    def test_get_signals_raises_not_implemented(self) -> None:
        s = EventDrivenStrategy()
        with pytest.raises(NotImplementedError):
            s.get_signals(pd.DataFrame())

    def test_get_signals_error_mentions_class_name(self) -> None:
        class MySuperStrategy(EventDrivenStrategy):
            pass

        s = MySuperStrategy()
        with pytest.raises(NotImplementedError, match="MySuperStrategy"):
            s.get_signals(pd.DataFrame())

    def test_is_not_vectorized(self) -> None:
        assert EventDrivenStrategy()._is_vectorized() is False

    def test_zero_warmup_default(self) -> None:
        s = EventDrivenStrategy()
        assert s.warmup_period == 0
        assert s.bar_index == 0


# ---------------------------------------------------------------------------
# VectorizedStrategy
# ---------------------------------------------------------------------------


class _ConcreteVectorized(VectorizedStrategy):
    def get_signals(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=data.index)


class TestVectorizedStrategy:
    def test_on_bar_raises_not_implemented(self) -> None:
        s = _ConcreteVectorized()
        with pytest.raises(NotImplementedError):
            s.on_bar(_bar())

    def test_on_bar_error_mentions_class_name(self) -> None:
        s = _ConcreteVectorized()
        with pytest.raises(NotImplementedError, match="_ConcreteVectorized"):
            s.on_bar(_bar())

    def test_get_signals_base_raises_if_not_overridden(self) -> None:
        s = VectorizedStrategy()
        with pytest.raises(NotImplementedError):
            s.get_signals(pd.DataFrame())

    def test_overridden_get_signals_returns_series(self) -> None:
        s = _ConcreteVectorized()
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
        sig = s.get_signals(df)
        assert isinstance(sig, pd.Series)
        assert len(sig) == 3

    def test_is_vectorized(self) -> None:
        assert _ConcreteVectorized()._is_vectorized() is True

    def test_params_stored(self) -> None:
        class ParamStrat(VectorizedStrategy):
            def __init__(self, window: int = 10) -> None:
                super().__init__(window=window)

            def get_signals(self, data: pd.DataFrame) -> pd.Series:
                return pd.Series(dtype=float)

        s = ParamStrat(window=30)
        assert s.params == {"window": 30}


# ---------------------------------------------------------------------------
# Backtester construction validation
# ---------------------------------------------------------------------------


class TestBacktesterConstruction:
    def _make_all_stubs(self) -> dict:
        return dict(
            feed=_Feed(_make_bars()),
            strategy=EventDrivenStrategy(),
            execution=_Execution(),
            slippage=_Slippage(),
            commission=_Commission(),
            risk=_Risk(),
            reporter=_Reporter(),
        )

    def test_valid_construction_does_not_raise(self) -> None:
        Backtester(**self._make_all_stubs())

    def test_none_feed_raises_configuration_error(self) -> None:
        kwargs = self._make_all_stubs()
        kwargs["feed"] = None
        with pytest.raises(ConfigurationError, match="feed"):
            Backtester(**kwargs)

    def test_none_strategy_raises_configuration_error(self) -> None:
        kwargs = self._make_all_stubs()
        kwargs["strategy"] = None
        with pytest.raises(ConfigurationError, match="strategy"):
            Backtester(**kwargs)

    def test_wrong_type_raises_configuration_error(self) -> None:
        kwargs = self._make_all_stubs()
        kwargs["execution"] = "not-an-execution-model"
        with pytest.raises(ConfigurationError, match="execution"):
            Backtester(**kwargs)

    def test_all_missing_raises_configuration_error(self) -> None:
        for name in ("feed", "strategy", "execution", "slippage", "commission", "risk", "reporter"):
            kwargs = self._make_all_stubs()
            kwargs[name] = None
            with pytest.raises(ConfigurationError):
                Backtester(**kwargs)


# ---------------------------------------------------------------------------
# Backtester run() — event-driven lifecycle
# ---------------------------------------------------------------------------


class TestBacktesterRun:
    def test_run_returns_backtest_result(self) -> None:
        bt = _make_backtester(EventDrivenStrategy())
        result = bt.run()
        assert isinstance(result, BacktestResult)

    def test_on_start_called_once(self) -> None:
        s = _TrackingStrategy()
        _make_backtester(s).run()
        assert s.started is True

    def test_on_end_called_once(self) -> None:
        s = _TrackingStrategy()
        _make_backtester(s).run()
        assert s.ended is True

    def test_on_bar_called_for_every_bar(self) -> None:
        bars = _make_bars(7)
        s = _TrackingStrategy()
        _make_backtester(s, bars).run()
        assert len(s.bars_seen) == 7

    def test_warmup_bars_not_passed_to_on_bar(self) -> None:
        bars = _make_bars(10)
        s = _TrackingStrategy(warmup_period=4)
        _make_backtester(s, bars).run()
        # bars 1-4 skipped, bars 5-10 processed
        assert len(s.bars_seen) == 6

    def test_bar_index_is_correct_after_run(self) -> None:
        bars = _make_bars(5)
        s = _TrackingStrategy()
        _make_backtester(s, bars).run()
        assert s.bar_index == 5

    def test_strategy_params_in_result(self) -> None:
        s = EventDrivenStrategy(lookback=42)
        result = _make_backtester(s).run()
        assert result.params.get("lookback") == 42

    def test_strategy_name_in_result(self) -> None:
        s = _TrackingStrategy()
        result = _make_backtester(s).run()
        assert result.strategy_name == "_TrackingStrategy"

    def test_result_has_empty_equity_curve(self) -> None:
        result = _make_backtester(EventDrivenStrategy()).run()
        assert isinstance(result.equity_curve, pd.Series)
        assert result.equity_curve.empty

    def test_result_has_empty_trades(self) -> None:
        result = _make_backtester(EventDrivenStrategy()).run()
        assert isinstance(result.trades, pd.DataFrame)
        assert result.trades.empty

    def test_pending_orders_accumulated_from_strategy(self) -> None:
        """Orders enqueued by the strategy must be stored by the Backtester."""
        bars = _make_bars(3)
        s = _TrackingStrategy()  # submits 1 buy per bar
        bt = _make_backtester(s, bars)
        bt.run()
        # 3 bars → 3 orders pending (execution is Phase 4)
        assert len(bt._pending_orders) == 3

    def test_current_bar_set_before_on_bar(self) -> None:
        seen_bars: list[Optional[Bar]] = []

        class BarSnapshotStrategy(EventDrivenStrategy):
            def on_bar(self, bar: Bar, context: Any = None) -> None:
                seen_bars.append(self.current_bar)

        bars = _make_bars(3)
        _make_backtester(BarSnapshotStrategy(), bars).run()

        assert len(seen_bars) == 3
        for i, b in enumerate(seen_bars):
            assert b is bars[i]

    def test_bar_event_emitted_per_bar(self) -> None:
        events: list[Event] = []
        bars = _make_bars(4)

        s = EventDrivenStrategy()
        bt = _make_backtester(s, bars)
        bt._bus.subscribe(EventType.BAR, events.append)
        bt.run()

        assert len(events) == 4
        assert all(e.type == EventType.BAR for e in events)

    def test_empty_feed_runs_without_calling_on_bar(self) -> None:
        s = _TrackingStrategy()
        bt = _make_backtester(s, [])
        bt.run()
        assert s.bars_seen == []
        assert s.started is True
        assert s.ended is True

    def test_on_start_error_raises_strategy_error(self) -> None:
        class BadStart(EventDrivenStrategy):
            def on_start(self, context: Any = None) -> None:
                raise RuntimeError("start failed")

        with pytest.raises(StrategyError, match="on_start"):
            _make_backtester(BadStart()).run()

    def test_on_end_error_raises_strategy_error(self) -> None:
        class BadEnd(EventDrivenStrategy):
            def on_end(self, context: Any = None) -> None:
                raise RuntimeError("end failed")

        with pytest.raises(StrategyError, match="on_end"):
            _make_backtester(BadEnd()).run()

    def test_on_bar_error_raises_strategy_error(self) -> None:
        class BadBar(EventDrivenStrategy):
            def on_bar(self, bar: Bar, context: Any = None) -> None:
                raise RuntimeError("bar failed")

        with pytest.raises(StrategyError, match="on_bar"):
            _make_backtester(BadBar(), _make_bars(1)).run()


# ---------------------------------------------------------------------------
# Backtester debug mode
# ---------------------------------------------------------------------------


class TestBacktesterDebugMode:
    def test_debug_mode_logs_bars(self) -> None:
        bars = _make_bars(3)
        bt = _make_backtester(EventDrivenStrategy(), bars, debug=True)

        with patch("backtester.core.backtester.logger") as mock_log:
            bt.run()
            # At minimum there should be several debug calls
            assert mock_log.debug.call_count >= 3  # start + 3 bars + finish

    def test_debug_false_does_not_log(self) -> None:
        bt = _make_backtester(EventDrivenStrategy(), debug=False)

        with patch("backtester.core.backtester.logger") as mock_log:
            bt.run()
            mock_log.debug.assert_not_called()
