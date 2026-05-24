"""
Concrete synchronous EventBus implementation.

Thread-safe via a threading.Lock so that background threads introduced in
later phases (e.g. the APScheduler-based run-launcher in the Streamlit UI)
can safely subscribe and emit without corrupting the handler registry.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable

from backtester.exceptions import BacktestError
from backtester.interfaces import Event, EventBus, EventType

logger = logging.getLogger(__name__)

# Type alias for clarity.
_Handler = Callable[[Event], None]


class SimpleEventBus(EventBus):
    """
    Synchronous publish/subscribe event dispatcher.

    Handlers are called in FIFO (first-subscribed, first-called) order per
    event type.  Dispatch is synchronous — ``emit`` blocks until all handlers
    for the event have been called or an exception is raised.

    Thread safety
    -------------
    A ``threading.Lock`` guards all reads and writes to the handler registry.
    The lock is released before handlers are invoked to prevent deadlock if a
    handler itself subscribes or emits.

    Error handling
    --------------
    If a handler raises an exception it is caught, logged at ERROR level, and
    re-raised wrapped in ``BacktestError``.  Handlers registered *after* the
    offending one are **not** called — callers of ``emit`` see the exception.
    """

    def __init__(self) -> None:
        # Map from EventType → ordered list of handler callables.
        self._handlers: dict[EventType, list[_Handler]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # EventBus interface
    # ------------------------------------------------------------------

    def subscribe(self, event_type: EventType, handler: _Handler) -> None:
        """
        Register *handler* to be called when *event_type* is emitted.

        Registering the same handler multiple times is allowed and results in
        multiple invocations per event.
        """
        with self._lock:
            self._handlers.setdefault(event_type, []).append(handler)

    def unsubscribe(self, event_type: EventType, handler: _Handler) -> None:
        """
        Remove the first occurrence of *handler* for *event_type*.

        Silently does nothing if the handler is not registered.
        """
        with self._lock:
            handlers = self._handlers.get(event_type)
            if handlers and handler in handlers:
                handlers.remove(handler)

    def emit(self, event: Event) -> None:
        """
        Dispatch *event* synchronously to all registered handlers in FIFO order.

        The handler list is copied while holding the lock and then released so
        that handlers can safely call ``subscribe`` or ``emit`` without
        deadlocking.

        Raises
        ------
        BacktestError
            If any handler raises an exception.  The original exception is
            chained via ``__cause__``.
        """
        with self._lock:
            # Snapshot so mutations during dispatch don't affect this cycle.
            handlers = list(self._handlers.get(event.type, []))

        for handler in handlers:
            try:
                handler(event)
            except BacktestError:
                # Already a domain error — log and propagate as-is.
                logger.error(
                    "BacktestError in handler %r for event %s",
                    handler,
                    event.type.name,
                )
                raise
            except Exception as exc:
                logger.error(
                    "Unexpected error in handler %r for event %s: %r",
                    handler,
                    event.type.name,
                    exc,
                )
                raise BacktestError(
                    f"Handler {handler!r} raised an unexpected error "
                    f"for event {event.type.name}: {exc}"
                ) from exc

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def subscriber_count(self, event_type: EventType) -> int:
        """Return the number of handlers registered for *event_type*."""
        with self._lock:
            return len(self._handlers.get(event_type, []))
