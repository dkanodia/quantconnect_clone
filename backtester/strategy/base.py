"""
BaseStrategy — concrete base class for all strategy implementations.

Provides no-op hook defaults and the internal/public API that the Backtester
relies on.  Users subclass EventDrivenStrategy or VectorizedStrategy; they
should not subclass BaseStrategy directly.

All types are imported from backtester.interfaces and backtester.exceptions
only — no cross-sibling imports.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from backtester.exceptions import InvalidOrderError
from backtester.interfaces import (
    Bar,
    DataFeed,
    EventBus,
    Order,
    OrderSide,
    OrderType,
    Strategy,
)


class BaseStrategy(Strategy):
    """
    Concrete base class for event-driven and vectorized strategies.

    Inherits the ``Strategy`` ABC and provides no-op implementations for all
    hook methods so that subclasses only need to override the hooks they use.

    Internal API (called by Backtester, not by user strategy code)
    ---------------------------------------------------------------
    ``_set_context(feed, portfolio, event_bus)``
        Injected by the Backtester before ``on_start`` is called.
    ``_update_current_bar(bar)``
        Sets ``_current_bar`` before each ``on_bar`` call.
    ``_enqueue_order(order)``
        Appends an ``Order`` to the pending queue.
    ``_flush_orders() -> list[Order]``
        Returns and clears the pending order queue.
    ``_flush_cancellations() -> list[str]``
        Returns and clears the pending cancellation queue (list of order IDs).
    ``_should_process_bar(bar_index: int) -> bool``
        Returns ``True`` if ``on_bar`` should be called for this bar.
        Overridden by ``EventDrivenStrategy`` to implement warmup logic.

    Public API (called by user strategy code inside ``on_bar`` etc.)
    ----------------------------------------------------------------
    ``self.buy(...)`` / ``self.sell(...)``
        Construct and enqueue a ``BUY`` / ``SELL`` Order.  Return the
        ``order_id`` string so the strategy can cancel it later.
    ``self.cancel(order_id)``
        Enqueue a cancellation request for the given ``order_id``.
    ``self.current_bar``
        The ``Bar`` currently being processed (set by Backtester).
    ``self.params``
        Dict of constructor keyword arguments (strategy parameters).

    Parameters
    ----------
    **params
        Arbitrary keyword arguments stored in ``self.params``.
    """

    def __init__(self, **params: Any) -> None:
        self.params: dict[str, Any] = dict(params)

        # Set by _set_context before on_start.
        self._feed: Optional[DataFeed] = None
        self._portfolio: Any = None          # PortfolioTracker in Phase 4
        self._event_bus: Optional[EventBus] = None

        # Set by Backtester before each on_bar call.
        self._current_bar: Optional[Bar] = None

        # Order / cancellation queues — flushed at the end of each bar.
        self._pending_orders: list[Order] = []
        self._pending_cancellations: list[str] = []

    # ------------------------------------------------------------------
    # Internal API (called by Backtester)
    # ------------------------------------------------------------------

    def _set_context(
        self,
        feed: DataFeed,
        portfolio: Any,
        event_bus: EventBus,
    ) -> None:
        """
        Inject runtime context before the backtest starts.

        Called once by the Backtester immediately before ``on_start``.
        Subclasses that need access to portfolio or event_bus in their hooks
        can access them via ``self._portfolio`` and ``self._event_bus``.
        """
        self._feed = feed
        self._portfolio = portfolio
        self._event_bus = event_bus

    def _update_current_bar(self, bar: Bar) -> None:
        """Set the bar that is currently being processed."""
        self._current_bar = bar

    def _enqueue_order(self, order: Order) -> None:
        """Append *order* to the pending order queue."""
        self._pending_orders.append(order)

    def _flush_orders(self) -> list[Order]:
        """Return and clear the pending order queue."""
        pending = list(self._pending_orders)
        self._pending_orders.clear()
        return pending

    def _flush_cancellations(self) -> list[str]:
        """Return and clear the pending cancellation queue (list of order IDs)."""
        pending = list(self._pending_cancellations)
        self._pending_cancellations.clear()
        return pending

    def _should_process_bar(self, bar_index: int) -> bool:
        """
        Return ``True`` if ``on_bar`` should be called for this bar.

        The default implementation always returns ``True``.
        ``EventDrivenStrategy`` overrides this to skip warmup bars.

        Parameters
        ----------
        bar_index:
            One-based index of the current bar (1 = first bar seen).
        """
        return True

    def _is_vectorized(self) -> bool:
        """Return ``True`` if this strategy operates in vectorized mode."""
        return False

    # ------------------------------------------------------------------
    # Public strategy API (called from user-overridden hooks)
    # ------------------------------------------------------------------

    @property
    def current_bar(self) -> Optional[Bar]:
        """The ``Bar`` currently being processed, or ``None`` before the loop starts."""
        return self._current_bar

    def buy(
        self,
        symbol: str,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> str:
        """
        Enqueue a BUY order and return its ``order_id``.

        Parameters
        ----------
        symbol:      Ticker to buy.
        quantity:    Number of shares/contracts.
        order_type:  Defaults to ``MARKET``.
        limit_price: Required for ``LIMIT`` and ``STOP_LIMIT`` orders.
        stop_price:  Required for ``STOP`` and ``STOP_LIMIT`` orders.

        Raises
        ------
        InvalidOrderError
            If the order parameters are structurally invalid (e.g. ``LIMIT``
            order without a ``limit_price``).
        """
        try:
            order = Order(
                symbol=symbol,
                side=OrderSide.BUY,
                type=order_type,
                quantity=quantity,
                limit_price=limit_price,
                stop_price=stop_price,
            )
        except ValueError as exc:
            raise InvalidOrderError(str(exc), symbol=symbol) from exc

        self._enqueue_order(order)
        return order.id

    def sell(
        self,
        symbol: str,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> str:
        """
        Enqueue a SELL order and return its ``order_id``.

        Parameters
        ----------
        symbol:      Ticker to sell.
        quantity:    Number of shares/contracts.
        order_type:  Defaults to ``MARKET``.
        limit_price: Required for ``LIMIT`` and ``STOP_LIMIT`` orders.
        stop_price:  Required for ``STOP`` and ``STOP_LIMIT`` orders.

        Raises
        ------
        InvalidOrderError
            If the order parameters are structurally invalid.
        """
        try:
            order = Order(
                symbol=symbol,
                side=OrderSide.SELL,
                type=order_type,
                quantity=quantity,
                limit_price=limit_price,
                stop_price=stop_price,
            )
        except ValueError as exc:
            raise InvalidOrderError(str(exc), symbol=symbol) from exc

        self._enqueue_order(order)
        return order.id

    def cancel(self, order_id: str) -> None:
        """
        Enqueue a cancellation request for the order with the given ``order_id``.

        The cancellation is processed by the Backtester at the end of the
        current bar (Phase 4 and later).
        """
        self._pending_cancellations.append(order_id)

    # ------------------------------------------------------------------
    # Strategy ABC — no-op defaults
    # ------------------------------------------------------------------

    def on_start(self, context: Any = None) -> None:
        """Called once before the first bar.  Override to initialise state."""

    def on_bar(self, bar: Bar, context: Any = None) -> None:
        """Called on every bar (after warmup).  Override to generate signals."""

    def on_order(self, order: Order, context: Any = None) -> None:
        """Called when an order status changes.  Override to react to fills."""

    def on_end(self, context: Any = None) -> None:
        """Called once after the last bar.  Override for cleanup."""

    def get_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Return a signal Series for vectorized mode.

        The default returns an empty Series.  ``EventDrivenStrategy``
        overrides this to raise ``NotImplementedError``.
        """
        return pd.Series(dtype=float)
