"""
Phase 3 tests — SimpleEventBus.

Tests cover:
- subscribe / unsubscribe / emit mechanics
- FIFO dispatch order
- handler exception behaviour (logged + re-raised as BacktestError)
- thread-safety under concurrent subscribe and emit
"""

from __future__ import annotations

import threading
import time
from typing import Any
from unittest.mock import patch

import pytest

from backtester.core.event_bus import SimpleEventBus
from backtester.exceptions import BacktestError
from backtester.interfaces import Event, EventType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bar_event() -> Event:
    from datetime import datetime, timezone
    return Event(type=EventType.BAR, payload="test_bar", timestamp=datetime.now(timezone.utc))


def _fill_event() -> Event:
    from datetime import datetime, timezone
    return Event(type=EventType.FILL, payload="test_fill", timestamp=datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Basic subscribe / unsubscribe / emit
# ---------------------------------------------------------------------------


class TestBasicBehaviour:
    def test_subscribe_and_emit_delivers_event(self) -> None:
        bus = SimpleEventBus()
        received: list[Event] = []

        bus.subscribe(EventType.BAR, received.append)
        bus.emit(_bar_event())

        assert len(received) == 1

    def test_unsubscribed_handler_not_called(self) -> None:
        bus = SimpleEventBus()
        received: list[Event] = []

        bus.subscribe(EventType.BAR, received.append)
        bus.unsubscribe(EventType.BAR, received.append)
        bus.emit(_bar_event())

        assert received == []

    def test_emit_with_no_subscribers_is_a_noop(self) -> None:
        bus = SimpleEventBus()
        # Should not raise.
        bus.emit(_bar_event())

    def test_unsubscribe_unknown_handler_is_a_noop(self) -> None:
        bus = SimpleEventBus()
        # Should not raise.
        bus.unsubscribe(EventType.BAR, lambda e: None)

    def test_handler_receives_correct_event(self) -> None:
        bus = SimpleEventBus()
        received: list[Event] = []

        bus.subscribe(EventType.FILL, received.append)
        event = _fill_event()
        bus.emit(event)

        assert received[0] is event

    def test_event_types_are_independent(self) -> None:
        """Handlers for one event type must not receive other types."""
        bus = SimpleEventBus()
        bar_calls: list[Event] = []
        fill_calls: list[Event] = []

        bus.subscribe(EventType.BAR, bar_calls.append)
        bus.subscribe(EventType.FILL, fill_calls.append)

        bus.emit(_bar_event())

        assert len(bar_calls) == 1
        assert fill_calls == []

    def test_multiple_handlers_for_same_event(self) -> None:
        bus = SimpleEventBus()
        calls: list[str] = []

        bus.subscribe(EventType.BAR, lambda e: calls.append("a"))
        bus.subscribe(EventType.BAR, lambda e: calls.append("b"))
        bus.emit(_bar_event())

        assert calls == ["a", "b"]

    def test_same_handler_registered_twice_called_twice(self) -> None:
        bus = SimpleEventBus()
        count: list[int] = [0]

        def handler(e: Event) -> None:
            count[0] += 1

        bus.subscribe(EventType.BAR, handler)
        bus.subscribe(EventType.BAR, handler)
        bus.emit(_bar_event())

        assert count[0] == 2

    def test_subscriber_count(self) -> None:
        bus = SimpleEventBus()
        assert bus.subscriber_count(EventType.BAR) == 0

        bus.subscribe(EventType.BAR, lambda e: None)
        assert bus.subscriber_count(EventType.BAR) == 1

        bus.subscribe(EventType.BAR, lambda e: None)
        assert bus.subscriber_count(EventType.BAR) == 2

    def test_unsubscribe_removes_only_first_occurrence(self) -> None:
        bus = SimpleEventBus()
        calls: list[int] = []
        handler = lambda e: calls.append(1)

        bus.subscribe(EventType.BAR, handler)
        bus.subscribe(EventType.BAR, handler)
        bus.unsubscribe(EventType.BAR, handler)  # removes ONE occurrence
        bus.emit(_bar_event())

        assert calls == [1]  # still called once


# ---------------------------------------------------------------------------
# FIFO dispatch order
# ---------------------------------------------------------------------------


class TestFIFODispatchOrder:
    def test_handlers_called_in_subscription_order(self) -> None:
        bus = SimpleEventBus()
        order: list[int] = []

        bus.subscribe(EventType.BAR, lambda e: order.append(1))
        bus.subscribe(EventType.BAR, lambda e: order.append(2))
        bus.subscribe(EventType.BAR, lambda e: order.append(3))
        bus.emit(_bar_event())

        assert order == [1, 2, 3]

    def test_multiple_emits_each_call_handlers_in_order(self) -> None:
        bus = SimpleEventBus()
        order: list[int] = []

        bus.subscribe(EventType.BAR, lambda e: order.append(1))
        bus.subscribe(EventType.BAR, lambda e: order.append(2))

        bus.emit(_bar_event())
        bus.emit(_bar_event())

        assert order == [1, 2, 1, 2]


# ---------------------------------------------------------------------------
# Exception handling
# ---------------------------------------------------------------------------


class TestExceptionHandling:
    def test_handler_exception_reraises_as_backtest_error(self) -> None:
        bus = SimpleEventBus()

        def bad_handler(event: Event) -> None:
            raise ValueError("handler failed")

        bus.subscribe(EventType.BAR, bad_handler)

        with pytest.raises(BacktestError):
            bus.emit(_bar_event())

    def test_handler_exception_is_chained(self) -> None:
        bus = SimpleEventBus()
        original = ValueError("root cause")

        bus.subscribe(EventType.BAR, lambda e: (_ for _ in ()).throw(original))

        with pytest.raises(BacktestError) as exc_info:
            bus.emit(_bar_event())

        assert exc_info.value.__cause__ is original

    def test_backtest_error_from_handler_propagates_unchanged(self) -> None:
        """A handler that raises BacktestError should not be double-wrapped."""
        bus = SimpleEventBus()
        original = BacktestError("domain error")

        bus.subscribe(EventType.BAR, lambda e: (_ for _ in ()).throw(original))

        with pytest.raises(BacktestError) as exc_info:
            bus.emit(_bar_event())

        assert exc_info.value is original

    def test_exception_is_logged(self) -> None:
        bus = SimpleEventBus()

        def bad_handler(event: Event) -> None:
            raise RuntimeError("boom")

        bus.subscribe(EventType.BAR, bad_handler)

        with patch("backtester.core.event_bus.logger") as mock_log:
            with pytest.raises(BacktestError):
                bus.emit(_bar_event())
            mock_log.error.assert_called_once()

    def test_second_handler_not_called_after_first_raises(self) -> None:
        bus = SimpleEventBus()
        second_called: list[bool] = []

        bus.subscribe(EventType.BAR, lambda e: (_ for _ in ()).throw(RuntimeError("bad")))
        bus.subscribe(EventType.BAR, lambda e: second_called.append(True))

        with pytest.raises(BacktestError):
            bus.emit(_bar_event())

        assert second_called == []


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_subscribe_does_not_corrupt_registry(self) -> None:
        """Many threads subscribing simultaneously should not cause data races."""
        bus = SimpleEventBus()
        n_threads = 50
        n_events_per_thread = 10
        total_calls: list[int] = []
        lock = threading.Lock()

        def subscribe_and_emit() -> None:
            def handler(e: Event) -> None:
                with lock:
                    total_calls.append(1)

            bus.subscribe(EventType.BAR, handler)
            for _ in range(n_events_per_thread):
                bus.emit(_bar_event())

        threads = [threading.Thread(target=subscribe_and_emit) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # We can't predict exact call counts due to race windows, but there
        # must be no exception and at least some calls.
        assert len(total_calls) > 0

    def test_emit_from_multiple_threads_is_safe(self) -> None:
        """Multiple threads emitting the same event concurrently must not crash."""
        bus = SimpleEventBus()
        calls: list[int] = []
        lock = threading.Lock()

        def counter(e: Event) -> None:
            with lock:
                calls.append(1)

        bus.subscribe(EventType.BAR, counter)

        def emit_many() -> None:
            for _ in range(20):
                bus.emit(_bar_event())

        threads = [threading.Thread(target=emit_many) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(calls) == 200
