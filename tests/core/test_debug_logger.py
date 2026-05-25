"""
Tests for DebugLogger.

Covers:
- Disabled logger emits nothing
- Each log_* method emits exactly one record at DEBUG level
- Every record deserialises as valid JSON with the correct "event" key
- Monetary values are rounded to 4 decimal places in output
- Logger name matches backtester.debug.<strategy_name>
- log_phase emits "event": "phase" with a "phase" key
- log_risk_check includes "approved" and optional "reason"
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Generator

import pytest

from backtester.core.debug_logger import DebugLogger
from backtester.interfaces import Bar, Event, EventType, Fill, Order, OrderSide, OrderType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Capture(logging.Handler):
    """In-memory log handler that stores every LogRecord."""

    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def messages(self) -> list[str]:
        return [self.format(r) for r in self.records]


@contextmanager
def _capture(logger_name: str) -> Generator[_Capture, None, None]:
    """Attach a _Capture handler to *logger_name*, yield it, then detach."""
    log = logging.getLogger(logger_name)
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
_TS = datetime(2024, 1, 15, tzinfo=_UTC)
_STRAT = "TestStrategy"


def _bar(close: float = 100.0) -> Bar:
    return Bar(
        symbol="SPY",
        timestamp=_TS,
        open=99.1234567,
        high=101.9876543,
        low=98.0,
        close=close,
        volume=1_000_000.123456,
    )


def _order(qty: float = 10.0) -> Order:
    return Order(symbol="SPY", side=OrderSide.BUY, type=OrderType.MARKET, quantity=qty)


def _fill(price: float = 100.0, commission: float = 1.0) -> Fill:
    ord_ = _order()
    return Fill(
        order_id=ord_.id,
        symbol="SPY",
        side=OrderSide.BUY,
        quantity=10.0,
        price=price,
        commission=commission,
        timestamp=_TS,
    )


def _event() -> Event:
    return Event(type=EventType.BAR, payload="test", timestamp=_TS)


def _make_logger(enabled: bool = True) -> DebugLogger:
    return DebugLogger(strategy_name=_STRAT, enabled=enabled)


def _one_record(handler: _Capture) -> dict:
    assert len(handler.records) == 1, (
        f"Expected exactly 1 record, got {len(handler.records)}"
    )
    raw = handler.records[0].getMessage()
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Logger identity
# ---------------------------------------------------------------------------


class TestLoggerIdentity:
    def test_logger_name_matches_strategy(self) -> None:
        dl = _make_logger()
        assert dl._logger.name == f"backtester.debug.{_STRAT}"

    def test_different_strategy_names_create_distinct_loggers(self) -> None:
        a = DebugLogger("AlphaStrategy")
        b = DebugLogger("BetaStrategy")
        assert a._logger.name != b._logger.name


# ---------------------------------------------------------------------------
# Disabled logger emits nothing
# ---------------------------------------------------------------------------


class TestDisabledLogger:
    def test_disabled_log_bar_emits_nothing(self) -> None:
        dl = DebugLogger(_STRAT, enabled=False)
        with _capture(f"backtester.debug.{_STRAT}") as h:
            dl.log_bar(_bar(), 1)
        assert h.records == []

    def test_disabled_log_event_emits_nothing(self) -> None:
        dl = DebugLogger(_STRAT, enabled=False)
        with _capture(f"backtester.debug.{_STRAT}") as h:
            dl.log_event(_event())
        assert h.records == []

    def test_disabled_all_methods_emit_nothing(self) -> None:
        dl = DebugLogger(_STRAT, enabled=False)
        with _capture(f"backtester.debug.{_STRAT}") as h:
            dl.log_bar(_bar(), 1)
            dl.log_event(_event())
            dl.log_order_submitted(_order())
            dl.log_order_cancelled(_order(), "reason")
            dl.log_order_rejected(_order(), "reason")
            dl.log_fill(_fill())
            dl.log_risk_check(_order(), approved=True)
            dl.log_equity(_TS, 100_000.0)
            dl.log_phase("on_start")
        assert h.records == []


# ---------------------------------------------------------------------------
# Each method emits exactly one record
# ---------------------------------------------------------------------------


class TestOneRecordPerCall:
    def test_log_bar_emits_one_record(self) -> None:
        dl = _make_logger()
        with _capture(f"backtester.debug.{_STRAT}") as h:
            dl.log_bar(_bar(), 1)
        assert len(h.records) == 1

    def test_log_event_emits_one_record(self) -> None:
        dl = _make_logger()
        with _capture(f"backtester.debug.{_STRAT}") as h:
            dl.log_event(_event())
        assert len(h.records) == 1

    def test_log_order_submitted_emits_one_record(self) -> None:
        dl = _make_logger()
        with _capture(f"backtester.debug.{_STRAT}") as h:
            dl.log_order_submitted(_order())
        assert len(h.records) == 1

    def test_log_order_cancelled_emits_one_record(self) -> None:
        dl = _make_logger()
        with _capture(f"backtester.debug.{_STRAT}") as h:
            dl.log_order_cancelled(_order(), "test reason")
        assert len(h.records) == 1

    def test_log_order_rejected_emits_one_record(self) -> None:
        dl = _make_logger()
        with _capture(f"backtester.debug.{_STRAT}") as h:
            dl.log_order_rejected(_order(), "test reason")
        assert len(h.records) == 1

    def test_log_fill_emits_one_record(self) -> None:
        dl = _make_logger()
        with _capture(f"backtester.debug.{_STRAT}") as h:
            dl.log_fill(_fill())
        assert len(h.records) == 1

    def test_log_risk_check_emits_one_record(self) -> None:
        dl = _make_logger()
        with _capture(f"backtester.debug.{_STRAT}") as h:
            dl.log_risk_check(_order(), approved=True)
        assert len(h.records) == 1

    def test_log_equity_emits_one_record(self) -> None:
        dl = _make_logger()
        with _capture(f"backtester.debug.{_STRAT}") as h:
            dl.log_equity(_TS, 100_000.0)
        assert len(h.records) == 1

    def test_log_phase_emits_one_record(self) -> None:
        dl = _make_logger()
        with _capture(f"backtester.debug.{_STRAT}") as h:
            dl.log_phase("on_start")
        assert len(h.records) == 1


# ---------------------------------------------------------------------------
# Valid JSON with correct "event" key
# ---------------------------------------------------------------------------


class TestJsonContent:
    def test_log_bar_event_key(self) -> None:
        dl = _make_logger()
        with _capture(f"backtester.debug.{_STRAT}") as h:
            dl.log_bar(_bar(), 42)
        data = _one_record(h)
        assert data["event"] == "bar"
        assert data["bar_index"] == 42
        assert data["symbol"] == "SPY"

    def test_log_event_event_key(self) -> None:
        dl = _make_logger()
        with _capture(f"backtester.debug.{_STRAT}") as h:
            dl.log_event(_event())
        data = _one_record(h)
        assert data["event"] == "bus_event"
        assert data["event_type"] == "BAR"

    def test_log_order_submitted_event_key(self) -> None:
        dl = _make_logger()
        with _capture(f"backtester.debug.{_STRAT}") as h:
            dl.log_order_submitted(_order())
        data = _one_record(h)
        assert data["event"] == "order_submitted"
        assert data["side"] == "BUY"

    def test_log_order_cancelled_event_key(self) -> None:
        dl = _make_logger()
        with _capture(f"backtester.debug.{_STRAT}") as h:
            dl.log_order_cancelled(_order(), "strategy cancellation")
        data = _one_record(h)
        assert data["event"] == "order_cancelled"
        assert data["reason"] == "strategy cancellation"

    def test_log_order_rejected_event_key(self) -> None:
        dl = _make_logger()
        with _capture(f"backtester.debug.{_STRAT}") as h:
            dl.log_order_rejected(_order(), "risk limit")
        data = _one_record(h)
        assert data["event"] == "order_rejected"
        assert data["reason"] == "risk limit"

    def test_log_fill_event_key(self) -> None:
        dl = _make_logger()
        with _capture(f"backtester.debug.{_STRAT}") as h:
            dl.log_fill(_fill(price=105.0, commission=2.0))
        data = _one_record(h)
        assert data["event"] == "fill"
        assert data["price"] == pytest.approx(105.0)

    def test_log_risk_check_approved_true(self) -> None:
        dl = _make_logger()
        with _capture(f"backtester.debug.{_STRAT}") as h:
            dl.log_risk_check(_order(), approved=True)
        data = _one_record(h)
        assert data["event"] == "risk_check"
        assert data["approved"] is True

    def test_log_risk_check_approved_false_with_reason(self) -> None:
        dl = _make_logger()
        with _capture(f"backtester.debug.{_STRAT}") as h:
            dl.log_risk_check(_order(), approved=False, reason="drawdown exceeded")
        data = _one_record(h)
        assert data["approved"] is False
        assert data["reason"] == "drawdown exceeded"

    def test_log_risk_check_no_reason_when_approved(self) -> None:
        """When approved=True and reason='', the 'reason' key should be absent."""
        dl = _make_logger()
        with _capture(f"backtester.debug.{_STRAT}") as h:
            dl.log_risk_check(_order(), approved=True)
        data = _one_record(h)
        assert "reason" not in data

    def test_log_equity_event_key(self) -> None:
        dl = _make_logger()
        with _capture(f"backtester.debug.{_STRAT}") as h:
            dl.log_equity(_TS, 123_456.789)
        data = _one_record(h)
        assert data["event"] == "equity"
        assert "equity" in data

    def test_log_phase_event_and_phase_keys(self) -> None:
        dl = _make_logger()
        with _capture(f"backtester.debug.{_STRAT}") as h:
            dl.log_phase("on_end")
        data = _one_record(h)
        assert data["event"] == "phase"
        assert data["phase"] == "on_end"

    def test_all_records_are_valid_json(self) -> None:
        dl = _make_logger()
        with _capture(f"backtester.debug.{_STRAT}") as h:
            dl.log_bar(_bar(), 1)
            dl.log_event(_event())
            dl.log_order_submitted(_order())
            dl.log_fill(_fill())
            dl.log_equity(_TS, 100_000.0)
            dl.log_phase("on_start")
        for rec in h.records:
            data = json.loads(rec.getMessage())
            assert "event" in data


# ---------------------------------------------------------------------------
# Monetary rounding to 4 decimal places
# ---------------------------------------------------------------------------


class TestMonetaryRounding:
    def test_bar_close_rounded_to_4dp(self) -> None:
        dl = _make_logger()
        close = 412.123456789
        with _capture(f"backtester.debug.{_STRAT}") as h:
            dl.log_bar(_bar(close=close), 1)
        data = _one_record(h)
        assert data["close"] == round(close, 4)

    def test_bar_open_rounded(self) -> None:
        dl = _make_logger()
        with _capture(f"backtester.debug.{_STRAT}") as h:
            dl.log_bar(_bar(), 1)
        data = _one_record(h)
        assert data["open"] == pytest.approx(round(99.1234567, 4))

    def test_fill_price_rounded(self) -> None:
        dl = _make_logger()
        price = 100.123456789
        with _capture(f"backtester.debug.{_STRAT}") as h:
            dl.log_fill(_fill(price=price))
        data = _one_record(h)
        assert data["price"] == pytest.approx(round(price, 4))

    def test_fill_commission_rounded(self) -> None:
        dl = _make_logger()
        commission = 1.99999999
        with _capture(f"backtester.debug.{_STRAT}") as h:
            dl.log_fill(_fill(commission=commission))
        data = _one_record(h)
        assert data["commission"] == pytest.approx(round(commission, 4))

    def test_equity_rounded_to_4dp(self) -> None:
        dl = _make_logger()
        equity = 100_000.123456789
        with _capture(f"backtester.debug.{_STRAT}") as h:
            dl.log_equity(_TS, equity)
        data = _one_record(h)
        assert data["equity"] == pytest.approx(round(equity, 4))
