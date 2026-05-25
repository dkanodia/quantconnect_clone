"""
DebugLogger — structured, zero-overhead debug logging for the backtesting engine.

Every log line is a single JSON object emitted at ``DEBUG`` level through
Python's standard ``logging`` module.  When ``enabled=False`` every method
is a no-op so production runs incur zero overhead.

Logger naming: ``backtester.debug.<strategy_name>``

Log format example
------------------
``{"event": "bar", "bar_index": 42, "symbol": "SPY", "close": 412.5, "timestamp": "2023-01-15T00:00:00+00:00"}``
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from backtester.interfaces import Bar, Event, Fill, Order


class DebugLogger:
    """
    Structured JSON debug logger for a single backtest run.

    All monetary / price values are rounded to 4 decimal places before
    serialisation so log files remain diff-friendly across minor floating-point
    variations.

    Parameters
    ----------
    strategy_name : str
        Used as the leaf segment of the Python logger name:
        ``backtester.debug.<strategy_name>``.
    enabled : bool
        When ``False`` every method is a no-op.  Pass ``debug`` from
        ``Backtester.__init__`` so production runs have zero overhead.
    """

    def __init__(self, strategy_name: str, enabled: bool = True) -> None:
        self._enabled = enabled
        self._logger = logging.getLogger(f"backtester.debug.{strategy_name}")

    # ------------------------------------------------------------------
    # Public logging methods
    # ------------------------------------------------------------------

    def log_bar(self, bar: Bar, bar_index: int) -> None:
        """Emit one record for a bar received from the DataFeed."""
        if not self._enabled:
            return
        self._emit(
            {
                "event": "bar",
                "bar_index": bar_index,
                "symbol": bar.symbol,
                "open": round(bar.open, 4),
                "high": round(bar.high, 4),
                "low": round(bar.low, 4),
                "close": round(bar.close, 4),
                "volume": round(bar.volume, 4),
                "timestamp": bar.timestamp.isoformat(),
            }
        )

    def log_event(self, event: Event) -> None:
        """Emit one record each time the EventBus dispatches an event."""
        if not self._enabled:
            return
        self._emit(
            {
                "event": "bus_event",
                "event_type": event.type.name,
                "timestamp": event.timestamp.isoformat(),
            }
        )

    def log_order_submitted(self, order: Order) -> None:
        """Emit one record when an order is submitted to the OrderManager."""
        if not self._enabled:
            return
        self._emit(
            {
                "event": "order_submitted",
                "order_id": order.id,
                "symbol": order.symbol,
                "side": order.side.value,
                "order_type": order.type.value,
                "quantity": round(order.quantity, 4),
                "limit_price": round(order.limit_price, 4) if order.limit_price is not None else None,
                "stop_price": round(order.stop_price, 4) if order.stop_price is not None else None,
            }
        )

    def log_order_cancelled(self, order: Order, reason: str) -> None:
        """Emit one record when an order is cancelled (by the strategy or engine)."""
        if not self._enabled:
            return
        self._emit(
            {
                "event": "order_cancelled",
                "order_id": order.id,
                "symbol": order.symbol,
                "side": order.side.value,
                "quantity": round(order.quantity, 4),
                "reason": reason,
            }
        )

    def log_order_rejected(self, order: Order, reason: str) -> None:
        """Emit one record when a risk model rejects an order."""
        if not self._enabled:
            return
        self._emit(
            {
                "event": "order_rejected",
                "order_id": order.id,
                "symbol": order.symbol,
                "side": order.side.value,
                "quantity": round(order.quantity, 4),
                "reason": reason,
            }
        )

    def log_fill(self, fill: Fill) -> None:
        """Emit one record when an execution model produces a fill."""
        if not self._enabled:
            return
        self._emit(
            {
                "event": "fill",
                "order_id": fill.order_id,
                "symbol": fill.symbol,
                "side": fill.side.value,
                "quantity": round(fill.quantity, 4),
                "price": round(fill.price, 4),
                "commission": round(fill.commission, 4),
                "timestamp": fill.timestamp.isoformat(),
            }
        )

    def log_risk_check(
        self, order: Order, approved: bool, reason: str = ""
    ) -> None:
        """Emit one record for each order risk-checked, showing approval outcome."""
        if not self._enabled:
            return
        record: dict[str, Any] = {
            "event": "risk_check",
            "order_id": order.id,
            "symbol": order.symbol,
            "approved": approved,
        }
        if reason:
            record["reason"] = reason
        self._emit(record)

    def log_equity(self, timestamp: datetime, equity: float) -> None:
        """Emit one record each time the portfolio equity is snapshotted."""
        if not self._enabled:
            return
        self._emit(
            {
                "event": "equity",
                "equity": round(equity, 4),
                "timestamp": timestamp.isoformat(),
            }
        )

    def log_phase(self, phase: str) -> None:
        """
        Emit one record marking a high-level lifecycle phase.

        Parameters
        ----------
        phase : str
            E.g. ``"on_start"``, ``"on_end"``, ``"warmup"``.
        """
        if not self._enabled:
            return
        self._emit({"event": "phase", "phase": phase})

    # ------------------------------------------------------------------
    # Private helper
    # ------------------------------------------------------------------

    def _emit(self, record: dict[str, Any]) -> None:
        """Serialise *record* to JSON and emit at DEBUG level."""
        self._logger.debug(json.dumps(record))
