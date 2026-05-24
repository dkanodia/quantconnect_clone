"""
Phase 1 tests — interfaces.py and exceptions.py.

Each ABC is tested with a minimal concrete implementation to verify:
  - the interface can be subclassed,
  - the dataclasses and enums behave correctly,
  - the exception hierarchy and context fields work as specified.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Iterator, Optional, Sequence
from unittest.mock import MagicMock

import pandas as pd
import pytest

from backtester.exceptions import (
    BacktestError,
    ConfigurationError,
    DataFeedError,
    DataNotFoundError,
    ExecutionError,
    InsufficientFundsError,
    InsufficientPositionError,
    InvalidOrderError,
    InvalidSignalError,
    MaxDrawdownExceeded,
    NegativeCashError,
    OptimizationError,
    OrderRejectedError,
    PortfolioError,
    PositionSizeLimitExceeded,
    RiskModelError,
    StrategyError,
    WFOError,
)
from backtester.interfaces import (
    BacktestResult,
    Bar,
    CommissionModel,
    DataFeed,
    Event,
    EventBus,
    EventType,
    ExecutionModel,
    Fill,
    Optimizer,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    PortfolioTracker,
    Position,
    Reporter,
    RiskModel,
    SlippageModel,
    Strategy,
    Visibility,
)


# ---------------------------------------------------------------------------
# Helpers — minimal concrete implementations of every ABC
# ---------------------------------------------------------------------------


class _Feed(DataFeed):
    def __init__(self, bars: list[Bar]) -> None:
        self._bars = bars
        self._idx = 0

    def __iter__(self) -> Iterator[Bar]:
        self._idx = 0
        return self

    def __next__(self) -> Bar:
        if self._idx >= len(self._bars):
            raise StopIteration
        bar = self._bars[self._idx]
        self._idx += 1
        return bar

    def reset(self) -> None:
        self._idx = 0

    @property
    def symbols(self) -> list[str]:
        return list({b.symbol for b in self._bars})


class _Strategy(Strategy):
    def on_start(self, context: Any) -> None:
        pass

    def on_bar(self, bar: Bar, context: Any) -> None:
        pass

    def on_order(self, order: Order, context: Any) -> None:
        pass

    def on_end(self, context: Any) -> None:
        pass

    def get_signals(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=data.index)


class _Execution(ExecutionModel):
    def execute(self, order: Order, current_bar: Bar, next_bar: Optional[Bar]) -> Optional[Fill]:
        price = next_bar.open if next_bar else current_bar.close
        return Fill(
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=price,
            commission=0.0,
            timestamp=datetime.utcnow(),
        )


class _Slippage(SlippageModel):
    def adjust(self, raw_price: float, order: Order, bar: Bar) -> float:
        return raw_price


class _Commission(CommissionModel):
    def compute(self, fill: Fill) -> float:
        return 0.0


class _Portfolio(PortfolioTracker):
    def __init__(self) -> None:
        self._cash = 100_000.0
        self._positions: dict[str, Position] = {}
        self._equity: list[tuple[datetime, float]] = []
        self._trades: list[dict] = []

    def apply_fill(self, fill: Fill) -> None:
        cost = fill.price * fill.quantity + fill.commission
        if fill.side == OrderSide.BUY:
            self._cash -= cost
        else:
            self._cash += fill.price * fill.quantity - fill.commission

    def update_market(self, bars: Sequence[Bar]) -> None:
        pass

    def record_equity(self, timestamp: datetime) -> None:
        self._equity.append((timestamp, self._cash))

    def get_equity_curve(self) -> pd.Series:
        if not self._equity:
            return pd.Series(dtype=float)
        ts, vals = zip(*self._equity)
        return pd.Series(vals, index=pd.DatetimeIndex(ts))

    def get_positions(self) -> dict[str, Position]:
        return self._positions

    def get_cash(self) -> float:
        return self._cash

    def get_total_equity(self) -> float:
        return self._cash

    def get_trades(self) -> pd.DataFrame:
        return pd.DataFrame(self._trades)


class _Risk(RiskModel):
    def check(self, order: Order, portfolio: PortfolioTracker, current_bar: Bar) -> bool:
        return True

    def on_fill(self, fill: Fill, portfolio: PortfolioTracker) -> None:
        pass


class _Bus(EventBus):
    def __init__(self) -> None:
        self._handlers: dict[EventType, list] = {}

    def subscribe(self, event_type: EventType, handler: Any) -> None:
        self._handlers.setdefault(event_type, []).append(handler)

    def unsubscribe(self, event_type: EventType, handler: Any) -> None:
        self._handlers.get(event_type, []).remove(handler)

    def emit(self, event: Event) -> None:
        for h in self._handlers.get(event.type, []):
            h(event)


class _Optimizer(Optimizer):
    def optimize(
        self,
        strategy_class: type,
        param_space: dict[str, Any],
        objective: Any,
        n_trials: int = 100,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {k: v[0] for k, v in param_space.items()}


class _Reporter(Reporter):
    def report(self, result: BacktestResult) -> dict:
        return {"run_id": result.run_id}


# ---------------------------------------------------------------------------
# Dataclass and enum tests
# ---------------------------------------------------------------------------


class TestBar:
    def test_fields(self) -> None:
        ts = datetime(2024, 1, 2)
        bar = Bar("SPY", ts, 100.0, 105.0, 99.0, 103.0, 1_000_000)
        assert bar.symbol == "SPY"
        assert bar.open == 100.0
        assert bar.close == 103.0

    def test_is_dataclass(self) -> None:
        import dataclasses
        assert dataclasses.is_dataclass(Bar)


class TestOrder:
    def test_market_order_defaults(self) -> None:
        o = Order(symbol="AAPL", side=OrderSide.BUY, type=OrderType.MARKET, quantity=10)
        assert o.status == OrderStatus.CREATED
        assert o.limit_price is None
        assert len(o.id) == 36  # UUID4

    def test_limit_order_requires_limit_price(self) -> None:
        with pytest.raises(ValueError):
            Order(symbol="AAPL", side=OrderSide.BUY, type=OrderType.LIMIT, quantity=10)

    def test_stop_order_requires_stop_price(self) -> None:
        with pytest.raises(ValueError):
            Order(symbol="AAPL", side=OrderSide.BUY, type=OrderType.STOP, quantity=10)

    def test_stop_limit_requires_both(self) -> None:
        with pytest.raises(ValueError):
            Order(
                symbol="AAPL",
                side=OrderSide.BUY,
                type=OrderType.STOP_LIMIT,
                quantity=10,
                stop_price=95.0,
                # limit_price missing → should raise
            )

    def test_valid_limit_order(self) -> None:
        o = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            quantity=10,
            limit_price=150.0,
        )
        assert o.limit_price == 150.0


class TestFill:
    def test_fields(self) -> None:
        f = Fill(
            order_id="abc",
            symbol="SPY",
            side=OrderSide.BUY,
            quantity=5,
            price=400.0,
            commission=1.0,
            timestamp=datetime.utcnow(),
        )
        assert f.price == 400.0
        assert f.commission == 1.0


class TestPosition:
    def test_market_value(self) -> None:
        pos = Position(symbol="TSLA", quantity=10, avg_entry_price=200.0)
        assert pos.market_value(210.0) == pytest.approx(2100.0)


class TestBacktestResult:
    def test_defaults(self) -> None:
        result = BacktestResult(
            equity_curve=pd.Series([100_000, 101_000]),
            trades=pd.DataFrame(),
            metrics={},
            params={},
        )
        assert result.visibility == Visibility.PRIVATE
        assert len(result.run_id) == 36


class TestEnums:
    def test_order_side(self) -> None:
        assert OrderSide.BUY.value == "BUY"
        assert OrderSide.SELL.value == "SELL"

    def test_visibility(self) -> None:
        values = {v.value for v in Visibility}
        assert values == {"PRIVATE", "TEAM", "FEATURED"}

    def test_order_status_lifecycle(self) -> None:
        statuses = [s.value for s in OrderStatus]
        assert "CREATED" in statuses
        assert "FILLED" in statuses
        assert "CANCELLED" in statuses


# ---------------------------------------------------------------------------
# ABC implementation tests
# ---------------------------------------------------------------------------


class TestDataFeed:
    def _make_bars(self) -> list[Bar]:
        ts = datetime(2024, 1, 2)
        return [Bar("SPY", ts, 100 + i, 110 + i, 90 + i, 105 + i, 1e6) for i in range(5)]

    def test_iteration(self) -> None:
        bars = self._make_bars()
        feed = _Feed(bars)
        collected = list(feed)
        assert len(collected) == 5
        assert collected[0].open == 100.0

    def test_reset(self) -> None:
        bars = self._make_bars()
        feed = _Feed(bars)
        list(feed)
        feed.reset()
        assert list(feed)[0].open == 100.0

    def test_symbols(self) -> None:
        bars = self._make_bars()
        feed = _Feed(bars)
        assert feed.symbols == ["SPY"]


class TestStrategy:
    def test_hooks_callable(self) -> None:
        s = _Strategy()
        bar = Bar("SPY", datetime.utcnow(), 100, 110, 90, 105, 1e6)
        s.on_start(None)
        s.on_bar(bar, None)
        s.on_order(MagicMock(), None)
        s.on_end(None)

    def test_get_signals_returns_series(self) -> None:
        s = _Strategy()
        df = pd.DataFrame({"close": [1, 2, 3]})
        sig = s.get_signals(df)
        assert isinstance(sig, pd.Series)


class TestExecutionModel:
    def test_fill_on_next_bar(self) -> None:
        ex = _Execution()
        bar = Bar("SPY", datetime.utcnow(), 100, 110, 90, 105, 1e6)
        next_bar = Bar("SPY", datetime.utcnow(), 106, 112, 104, 109, 1e6)
        order = Order("SPY", OrderSide.BUY, OrderType.MARKET, 10)
        fill = ex.execute(order, bar, next_bar)
        assert fill is not None
        assert fill.price == next_bar.open

    def test_fill_at_close_when_no_next_bar(self) -> None:
        ex = _Execution()
        bar = Bar("SPY", datetime.utcnow(), 100, 110, 90, 105, 1e6)
        order = Order("SPY", OrderSide.BUY, OrderType.MARKET, 10)
        fill = ex.execute(order, bar, None)
        assert fill is not None
        assert fill.price == bar.close


class TestSlippageModel:
    def test_passthrough(self) -> None:
        s = _Slippage()
        bar = Bar("SPY", datetime.utcnow(), 100, 110, 90, 105, 1e6)
        order = Order("SPY", OrderSide.BUY, OrderType.MARKET, 1)
        assert s.adjust(100.0, order, bar) == 100.0


class TestCommissionModel:
    def test_zero_commission(self) -> None:
        c = _Commission()
        fill = Fill("x", "SPY", OrderSide.BUY, 10, 100.0, 0.0, datetime.utcnow())
        assert c.compute(fill) == 0.0


class TestPortfolioTracker:
    def test_initial_cash(self) -> None:
        p = _Portfolio()
        assert p.get_cash() == 100_000.0

    def test_equity_curve_grows(self) -> None:
        p = _Portfolio()
        ts = datetime.utcnow()
        p.record_equity(ts)
        curve = p.get_equity_curve()
        assert len(curve) == 1
        assert curve.iloc[0] == 100_000.0

    def test_apply_fill_reduces_cash(self) -> None:
        p = _Portfolio()
        fill = Fill("x", "SPY", OrderSide.BUY, 1, 200.0, 0.0, datetime.utcnow())
        p.apply_fill(fill)
        assert p.get_cash() == pytest.approx(99_800.0)


class TestRiskModel:
    def test_approve_order(self) -> None:
        r = _Risk()
        order = Order("SPY", OrderSide.BUY, OrderType.MARKET, 1)
        bar = Bar("SPY", datetime.utcnow(), 100, 110, 90, 105, 1e6)
        assert r.check(order, _Portfolio(), bar) is True


class TestEventBus:
    def test_subscribe_and_emit(self) -> None:
        bus = _Bus()
        received: list[Event] = []
        bus.subscribe(EventType.BAR, received.append)
        event = Event(EventType.BAR, payload="test")
        bus.emit(event)
        assert len(received) == 1
        assert received[0].payload == "test"

    def test_unsubscribe(self) -> None:
        bus = _Bus()
        received: list[Event] = []
        bus.subscribe(EventType.BAR, received.append)
        bus.unsubscribe(EventType.BAR, received.append)
        bus.emit(Event(EventType.BAR, payload="x"))
        assert len(received) == 0

    def test_multiple_handlers(self) -> None:
        bus = _Bus()
        calls: list[str] = []
        bus.subscribe(EventType.FILL, lambda e: calls.append("a"))
        bus.subscribe(EventType.FILL, lambda e: calls.append("b"))
        bus.emit(Event(EventType.FILL, payload=None))
        assert calls == ["a", "b"]


class TestOptimizer:
    def test_returns_best_params(self) -> None:
        opt = _Optimizer()
        result = opt.optimize(
            _Strategy,
            param_space={"lookback": [10, 20, 30]},
            objective=lambda cls, params: 0.0,
        )
        assert result == {"lookback": 10}


class TestReporter:
    def test_report(self) -> None:
        r = _Reporter()
        result = BacktestResult(
            equity_curve=pd.Series([100_000]),
            trades=pd.DataFrame(),
            metrics={},
            params={},
            run_id="test-run",
        )
        output = r.report(result)
        assert output["run_id"] == "test-run"


# ---------------------------------------------------------------------------
# Exception hierarchy tests
# ---------------------------------------------------------------------------


class TestExceptionHierarchy:
    def test_all_inherit_from_backtest_error(self) -> None:
        exc_classes = [
            DataFeedError,
            DataNotFoundError,
            StrategyError,
            InvalidSignalError,
            InvalidOrderError,
            OrderRejectedError,
            ExecutionError,
            InsufficientFundsError,
            InsufficientPositionError,
            PortfolioError,
            NegativeCashError,
            RiskModelError,
            MaxDrawdownExceeded,
            PositionSizeLimitExceeded,
            OptimizationError,
            WFOError,
            ConfigurationError,
        ]
        for cls in exc_classes:
            assert issubclass(cls, BacktestError), f"{cls.__name__} not a BacktestError subclass"

    def test_context_fields(self) -> None:
        err = BacktestError(
            "something went wrong",
            strategy_name="Momentum",
            symbol="SPY",
            bar_index=42,
            portfolio_snapshot={"cash": 50_000},
        )
        assert err.strategy_name == "Momentum"
        assert err.symbol == "SPY"
        assert err.bar_index == 42
        assert err.portfolio_snapshot["cash"] == 50_000

    def test_str_includes_context(self) -> None:
        err = BacktestError("oops", strategy_name="TestStrat", symbol="AAPL", bar_index=7)
        s = str(err)
        assert "TestStrat" in s
        assert "AAPL" in s
        assert "7" in s

    def test_order_rejected_error_fields(self) -> None:
        err = OrderRejectedError(
            "rejected",
            order_id="abc-123",
            reason="max drawdown",
            strategy_name="Trend",
        )
        assert err.order_id == "abc-123"
        assert err.reason == "max drawdown"
        s = str(err)
        assert "abc-123" in s
        assert "max drawdown" in s

    def test_insufficient_funds_error_fields(self) -> None:
        err = InsufficientFundsError("no cash", required=10_000.0, available=500.0)
        assert err.required == 10_000.0
        assert err.available == 500.0
        s = str(err)
        assert "10000.00" in s
        assert "500.00" in s

    def test_insufficient_position_error_fields(self) -> None:
        err = InsufficientPositionError("short", requested=100.0, held=50.0, symbol="SPY")
        assert err.requested == 100.0
        assert err.held == 50.0

    def test_max_drawdown_exceeded_fields(self) -> None:
        err = MaxDrawdownExceeded("dd", current_drawdown=0.20, limit=0.15)
        s = str(err)
        assert "20.00%" in s
        assert "15.00%" in s

    def test_position_size_limit_fields(self) -> None:
        err = PositionSizeLimitExceeded("size", requested_pct=0.30, limit_pct=0.10)
        s = str(err)
        assert "30.00%" in s
        assert "10.00%" in s

    def test_catch_as_backtest_error(self) -> None:
        with pytest.raises(BacktestError):
            raise MaxDrawdownExceeded("halt", current_drawdown=0.25, limit=0.15)
