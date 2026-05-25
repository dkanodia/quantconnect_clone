"""
OrderManager — tracks the lifecycle of all orders within a single backtest run.

Responsibilities:
- Accept order submissions and mark them SUBMITTED.
- Record fills and mark orders FILLED.
- Provide a snapshot of all SUBMITTED (pending) orders for the execution step.
- Support cancellation of orders that are not yet filled.

All raised exceptions are enriched with ``order_id``, ``symbol``, and
``strategy_name`` so diagnostics are self-contained.
"""

from __future__ import annotations

import logging
from typing import Optional

from backtester.exceptions import OrderError
from backtester.interfaces import Fill, Order, OrderStatus

logger = logging.getLogger(__name__)


class OrderManager:
    """
    Lifecycle manager for all orders in one backtest run.

    The Backtester creates a single ``OrderManager`` per run and routes every
    order through it.  It is **not** thread-safe by design — the event-driven
    loop is single-threaded.

    Parameters
    ----------
    strategy_name : str
        Name of the owning strategy.  Included in all raised exceptions to
        make error messages self-contained without needing a call stack.

    Methods
    -------
    submit(order)
        Mark an order as SUBMITTED and register it.  Raises ``OrderError`` if
        an order with the same id was already submitted.
    cancel(order_id)
        Mark a SUBMITTED order as CANCELLED.  Raises ``OrderError`` if the id
        is unknown, already FILLED, or already CANCELLED.
    record_fill(fill)
        Mark the associated order as FILLED and store the fill.  Raises
        ``OrderError`` if the order id is not registered.
    get_pending() → list[Order]
        Return all orders currently in SUBMITTED status.
    get_fills() → list[Fill]
        Return every fill recorded so far.
    get_order(order_id) → Order
        Look up an order by id.  Raises ``OrderError`` if not found.
    """

    def __init__(self, strategy_name: str = "") -> None:
        self._strategy_name = strategy_name
        self._orders: dict[str, Order] = {}
        self._fills: list[Fill] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, order: Order) -> None:
        """Register an order and set its status to SUBMITTED."""
        if order.id in self._orders:
            raise OrderError(
                f"Order '{order.id}' has already been submitted.",
                order_id=order.id,
                symbol=order.symbol,
                strategy_name=self._strategy_name,
            )
        order.status = OrderStatus.SUBMITTED
        self._orders[order.id] = order
        logger.debug(
            "Order submitted: id=%s symbol=%s side=%s qty=%s",
            order.id, order.symbol, order.side.value, order.quantity,
        )

    def cancel(self, order_id: str) -> None:
        """
        Cancel a SUBMITTED order.

        Raises
        ------
        OrderError
            If the order does not exist, is already FILLED, or is already
            CANCELLED.
        """
        order = self._orders.get(order_id)
        if order is None:
            raise OrderError(
                f"Order '{order_id}' not found — cannot cancel.",
                order_id=order_id,
                strategy_name=self._strategy_name,
            )
        if order.status == OrderStatus.FILLED:
            raise OrderError(
                f"Order '{order_id}' is already FILLED — cannot cancel.",
                order_id=order_id,
                symbol=order.symbol,
                strategy_name=self._strategy_name,
            )
        if order.status == OrderStatus.CANCELLED:
            raise OrderError(
                f"Order '{order_id}' is already CANCELLED.",
                order_id=order_id,
                symbol=order.symbol,
                strategy_name=self._strategy_name,
            )
        order.status = OrderStatus.CANCELLED
        logger.debug("Order cancelled: id=%s", order_id)

    def record_fill(self, fill: Fill) -> None:
        """
        Mark the order referenced by *fill* as FILLED and store the fill.

        Raises
        ------
        OrderError
            If the order id in the fill is not registered.
        """
        order = self._orders.get(fill.order_id)
        if order is None:
            raise OrderError(
                f"Cannot record fill: order '{fill.order_id}' not found.",
                order_id=fill.order_id,
                symbol=fill.symbol,
                strategy_name=self._strategy_name,
            )
        order.status = OrderStatus.FILLED
        self._fills.append(fill)
        logger.debug(
            "Fill recorded: order_id=%s symbol=%s price=%.4f qty=%s commission=%.4f",
            fill.order_id, fill.symbol, fill.price, fill.quantity, fill.commission,
        )

    def get_pending(self) -> list[Order]:
        """Return all orders in SUBMITTED status (snapshot, safe to iterate)."""
        return [o for o in self._orders.values() if o.status == OrderStatus.SUBMITTED]

    def get_fills(self) -> list[Fill]:
        """Return every fill recorded during this run."""
        return list(self._fills)

    def get_order(self, order_id: str) -> Order:
        """
        Retrieve an order by id.

        Raises
        ------
        OrderError
            If the id is not registered.
        """
        order = self._orders.get(order_id)
        if order is None:
            raise OrderError(
                f"Order '{order_id}' not found.",
                order_id=order_id,
                strategy_name=self._strategy_name,
            )
        return order
