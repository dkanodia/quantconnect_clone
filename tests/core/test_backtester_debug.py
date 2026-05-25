"""
Tests for Backtester debug mode and exception propagation policy (Phase 5).

Covers:
- debug=True emits structured JSON log events in the correct sequence
- debug=False emits zero DEBUG records
- Strategy hook errors wrapped as StrategyError with correct context fields
- RiskRejectionError (PositionSizeLimitExceeded) is caught: order cancelled,
  run continues, BacktestResult is valid
- MaxDrawdownExceeded propagates and halts the run
- _enrich backfills context into exceptions raised in the event loop
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Generator, Iterator, Optional, Sequence

import pytest

from backtester.core.backtester import Backtester
from backtester.exceptions import (
    MaxDrawdownExceeded,
    PositionSizeLimitExceeded,
    StrategyError,
)
from backtester.execution.commission_models import ZeroCommission
from backtester.execution.execution_models import NextOpenExecution, SameBarExecution
from backtester.execution.slippage_models import ZeroSlippage
from backtester.interfaces import (
    BacktestResult,
    Bar,
    CommissionModel,
    DataFeed,
    ExecutionModel,
    Fill,
    Order,
    OrderSide,
    OrderType,
    Reporter,
    RiskModel,
    SlippageModel,
    Strategy,
)
from backtester.risk.risk_models import MaxDrawdownHalt, NoRisk, PositionSizeLimit
from backtester.strategy.event_driven import EventDrivenStrategy


# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def json_messages(self) -> list[dict]:
        return [json.loads(r.getMessage()) for r in self.records]

    def event_sequence(self) -> list[str]:
        return [d["event"] for d in self.json_messages()]


@contextmanager
def _capture_debug(strategy_name: str) -> Generator[_Capture, None, None]:
    log = logging.getLogger(f"backtester.debug.{strategy_name}")
    handler = _Capture()
    old_level = log.level
    log.addHandler(handler)
    log.setLevel(logging.DEBUG)
    try:
        yield handler
    finally:
        log.removeHandler(handler)
        log.setLevel(old_level)


_UTC = timezone.utc
_ZERO_SLIP = ZeroSlippage()
_ZERO_COM = ZeroCommission()


def _bar(day: int, close: float = 100.0, symbol: str = "AAPL") -> Bar:
    return Bar(
        symbol=symbol,
        timestamp=datetime(2024, 1, day, tzinfo=_UTC),
        open=close,
        high=close + 5,
        low=close - 5,
        close=close,
        volume=10_000.0,
    )


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


class _Reporter(Reporter):
    def report(self, result: BacktestResult) -> None:
        pass


def _make_bt(
    bars: list[Bar],
    strategy: Strategy,
    risk: RiskModel = None,  # type: ignore[assignment]
    debug: bool = False,
    initial_cash: float = 100_000.0,
    execution: ExecutionModel = None,  # type: ignore[assignment]
) -> Backtester:
    return Backtester(
        feed=_Feed(bars),
        strategy=strategy,
        execution=execution or NextOpenExecution(_ZERO_SLIP, _ZERO_COM),
        slippage=_ZERO_SLIP,
        commission=_ZERO_COM,
        risk=risk or NoRisk(),
        reporter=_Reporter(),
        initial_cash=initial_cash,
        debug=debug,
    )


class _PassiveStrategy(EventDrivenStrategy):
    pass


class _BuyOnBar1(EventDrivenStrategy):
    def __init__(self) -> None:
        super().__init__()
        self._done = False

    def on_bar(self, bar: Bar, context: Any) -> None:
        if not self._done:
            self.buy(bar.symbol, quantity=10)
            self._done = True


# ---------------------------------------------------------------------------
# debug=False emits nothing
# ---------------------------------------------------------------------------


class TestDebugFalse:
    def test_debug_false_emits_no_debug_records(self) -> None:
        bars = [_bar(d) for d in range(2, 5)]
        bt = _make_bt(bars, _PassiveStrategy(), debug=False)
        with _capture_debug("_PassiveStrategy") as h:
            bt.run()
        assert len(h.records) == 0

    def test_debug_false_result_still_valid(self) -> None:
        bars = [_bar(d) for d in range(2, 4)]
        result = _make_bt(bars, _PassiveStrategy(), debug=False).run()
        assert isinstance(result, BacktestResult)


# ---------------------------------------------------------------------------
# debug=True emits JSON events
# ---------------------------------------------------------------------------


class TestDebugTrue:
    def test_debug_true_emits_phase_on_start(self) -> None:
        bars = [_bar(2)]
        bt = _make_bt(bars, _PassiveStrategy(), debug=True)
        with _capture_debug("_PassiveStrategy") as h:
            bt.run()
        events = h.event_sequence()
        assert "phase" in events
        phase_records = [d for d in h.json_messages() if d["event"] == "phase"]
        phases = [r["phase"] for r in phase_records]
        assert "on_start" in phases

    def test_debug_true_emits_phase_on_end(self) -> None:
        bars = [_bar(2)]
        bt = _make_bt(bars, _PassiveStrategy(), debug=True)
        with _capture_debug("_PassiveStrategy") as h:
            bt.run()
        phase_records = [d for d in h.json_messages() if d["event"] == "phase"]
        phases = [r["phase"] for r in phase_records]
        assert "on_end" in phases

    def test_debug_true_emits_bar_record_per_bar(self) -> None:
        n = 4
        bars = [_bar(d) for d in range(2, 2 + n)]
        bt = _make_bt(bars, _PassiveStrategy(), debug=True)
        with _capture_debug("_PassiveStrategy") as h:
            bt.run()
        bar_records = [d for d in h.json_messages() if d["event"] == "bar"]
        assert len(bar_records) == n

    def test_debug_true_emits_equity_record_per_bar(self) -> None:
        n = 3
        bars = [_bar(d) for d in range(2, 2 + n)]
        bt = _make_bt(bars, _PassiveStrategy(), debug=True)
        with _capture_debug("_PassiveStrategy") as h:
            bt.run()
        equity_records = [d for d in h.json_messages() if d["event"] == "equity"]
        assert len(equity_records) == n

    def test_debug_true_emits_bus_event_per_bar(self) -> None:
        bars = [_bar(d) for d in range(2, 5)]  # 3 bars
        bt = _make_bt(bars, _PassiveStrategy(), debug=True)
        with _capture_debug("_PassiveStrategy") as h:
            bt.run()
        bus_events = [d for d in h.json_messages() if d["event"] == "bus_event"]
        # At least one BAR bus event per bar
        bar_bus = [d for d in bus_events if d.get("event_type") == "BAR"]
        assert len(bar_bus) == 3

    def test_debug_true_emits_order_submitted_record(self) -> None:
        bars = [_bar(2), _bar(3), _bar(4)]
        bt = _make_bt(bars, _BuyOnBar1(), debug=True)
        with _capture_debug("_BuyOnBar1") as h:
            bt.run()
        submitted = [d for d in h.json_messages() if d["event"] == "order_submitted"]
        assert len(submitted) == 1

    def test_debug_true_emits_fill_and_risk_check_on_execution(self) -> None:
        bars = [_bar(2), _bar(3), _bar(4), _bar(5)]
        bt = _make_bt(bars, _BuyOnBar1(), debug=True)
        with _capture_debug("_BuyOnBar1") as h:
            bt.run()
        fills = [d for d in h.json_messages() if d["event"] == "fill"]
        risk_checks = [d for d in h.json_messages() if d["event"] == "risk_check"]
        assert len(fills) >= 1
        assert len(risk_checks) >= 1

    def test_debug_bar_records_contain_correct_symbol(self) -> None:
        bars = [_bar(2, symbol="MSFT")]
        bt = _make_bt(bars, _PassiveStrategy(), debug=True)
        with _capture_debug("_PassiveStrategy") as h:
            bt.run()
        bar_records = [d for d in h.json_messages() if d["event"] == "bar"]
        assert all(d["symbol"] == "MSFT" for d in bar_records)

    def test_debug_all_records_are_valid_json(self) -> None:
        bars = [_bar(d) for d in range(2, 5)]
        bt = _make_bt(bars, _BuyOnBar1(), debug=True)
        with _capture_debug("_BuyOnBar1") as h:
            bt.run()
        for rec in h.records:
            data = json.loads(rec.getMessage())
            assert "event" in data

    def test_on_start_phase_logged_before_first_bar(self) -> None:
        bars = [_bar(2)]
        bt = _make_bt(bars, _PassiveStrategy(), debug=True)
        with _capture_debug("_PassiveStrategy") as h:
            bt.run()
        events = h.event_sequence()
        on_start_idx = next(
            (i for i, e in enumerate(events)
             if e == "phase" and h.json_messages()[i].get("phase") == "on_start"),
            None,
        )
        first_bar_idx = next((i for i, e in enumerate(events) if e == "bar"), None)
        assert on_start_idx is not None
        assert first_bar_idx is not None
        assert on_start_idx < first_bar_idx


# ---------------------------------------------------------------------------
# Strategy hook error wrapping
# ---------------------------------------------------------------------------


class TestStrategyHookErrors:
    def test_on_bar_error_wrapped_as_strategy_error(self) -> None:
        class BadBar(EventDrivenStrategy):
            def on_bar(self, bar: Bar, context: Any) -> None:
                raise ValueError("bad logic")

        with pytest.raises(StrategyError) as exc_info:
            _make_bt([_bar(2)], BadBar()).run()
        assert exc_info.value.strategy_name == "BadBar"
        assert exc_info.value.bar_index == 1
        assert exc_info.value.symbol == "AAPL"

    def test_on_start_error_wrapped_as_strategy_error(self) -> None:
        class BadStart(EventDrivenStrategy):
            def on_start(self, context: Any) -> None:
                raise RuntimeError("startup error")

        with pytest.raises(StrategyError) as exc_info:
            _make_bt([_bar(2)], BadStart()).run()
        assert "on_start" in str(exc_info.value)
        assert exc_info.value.strategy_name == "BadStart"

    def test_on_end_error_wrapped_as_strategy_error(self) -> None:
        class BadEnd(EventDrivenStrategy):
            def on_end(self, context: Any) -> None:
                raise RuntimeError("teardown error")

        with pytest.raises(StrategyError) as exc_info:
            _make_bt([_bar(2)], BadEnd()).run()
        assert exc_info.value.strategy_name == "BadEnd"


# ---------------------------------------------------------------------------
# RiskRejectionError (soft rejection) — order cancelled, run continues
# ---------------------------------------------------------------------------


class _KeepBuying(EventDrivenStrategy):
    """Places one order per bar so risk checks fire repeatedly."""

    def __init__(self) -> None:
        super().__init__()
        self.on_order_calls: list[Order] = []

    def on_bar(self, bar: Bar, context: Any) -> None:
        self.buy(bar.symbol, quantity=1_000)  # large qty → exceeds position limit

    def on_order(self, order: Order, context: Any) -> None:
        self.on_order_calls.append(order)


class TestRiskRejectionContinues:
    def test_position_size_limit_does_not_halt_run(self) -> None:
        """PositionSizeLimitExceeded is a soft rejection — run must complete."""
        bars = [_bar(d) for d in range(2, 6)]
        strategy = _KeepBuying()
        # 1_000 shares @ 100 = 100_000 = 100% of equity >> 5% limit → rejected every bar
        risk = PositionSizeLimit(limit_pct=0.05)
        result = _make_bt(bars, strategy, risk=risk, initial_cash=100_000.0).run()
        assert isinstance(result, BacktestResult)

    def test_rejected_orders_produce_no_fills(self) -> None:
        """All risk-rejected orders should produce zero fills."""
        bars = [_bar(2), _bar(3), _bar(4)]
        strategy = _KeepBuying()
        risk = PositionSizeLimit(limit_pct=0.05)
        bt = _make_bt(bars, strategy, risk=risk, initial_cash=100_000.0)
        bt.run()
        # No fills should have been recorded (every order exceeded the limit)
        assert bt._order_manager.get_fills() == []

    def test_run_continues_and_returns_valid_result_after_rejection(self) -> None:
        bars = [_bar(d) for d in range(2, 7)]
        result = _make_bt(
            bars, _KeepBuying(),
            risk=PositionSizeLimit(limit_pct=0.01),
            initial_cash=100_000.0,
        ).run()
        assert result.metrics["num_trades"] == 0  # no fills since all rejected
        assert len(result.equity_curve) == len(bars)

    def test_rejection_logged_in_debug_mode(self) -> None:
        bars = [_bar(2), _bar(3), _bar(4)]
        strategy = _KeepBuying()
        risk = PositionSizeLimit(limit_pct=0.01)
        bt = _make_bt(bars, strategy, risk=risk, debug=True)
        with _capture_debug("_KeepBuying") as h:
            bt.run()
        rejected = [d for d in h.json_messages() if d["event"] == "order_rejected"]
        assert len(rejected) >= 1

    def test_on_order_not_called_for_rejected_orders(self) -> None:
        """strategy.on_order must only be called when a fill actually occurs."""
        bars = [_bar(2), _bar(3)]
        strategy = _KeepBuying()
        risk = PositionSizeLimit(limit_pct=0.01)
        _make_bt(bars, strategy, risk=risk).run()
        assert strategy.on_order_calls == []


# ---------------------------------------------------------------------------
# MaxDrawdownExceeded (hard halt) — run stops
# ---------------------------------------------------------------------------


class _BigBuyThenKeep(EventDrivenStrategy):
    def __init__(self) -> None:
        super().__init__()
        self._first = True

    def on_bar(self, bar: Bar, context: Any) -> None:
        if self._first:
            self.buy(bar.symbol, quantity=900)
            self._first = False
        else:
            self.buy(bar.symbol, quantity=1)


class TestMaxDrawdownHalt:
    def test_max_drawdown_exceeded_halts_run(self) -> None:
        bars = [
            _bar(2, close=100.0),
            _bar(3, close=100.0),
            _bar(4, close=10.0),   # collapse → big drawdown
            _bar(5, close=10.0),   # risk check fires here
        ]
        bt = _make_bt(bars, _BigBuyThenKeep(), risk=MaxDrawdownHalt(limit=0.01))
        with pytest.raises(MaxDrawdownExceeded):
            bt.run()

    def test_max_drawdown_exception_has_context(self) -> None:
        bars = [
            _bar(2, close=100.0),
            _bar(3, close=100.0),
            _bar(4, close=10.0),
            _bar(5, close=10.0),
        ]
        bt = _make_bt(bars, _BigBuyThenKeep(), risk=MaxDrawdownHalt(limit=0.01))
        with pytest.raises(MaxDrawdownExceeded) as exc_info:
            bt.run()
        exc = exc_info.value
        assert exc.current_drawdown is not None
        assert exc.current_drawdown > 0.01
        # _enrich should have backfilled bar_index and symbol
        assert exc.bar_index is not None
        assert exc.symbol is not None
