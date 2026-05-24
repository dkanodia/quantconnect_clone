"""
Backtester — orchestrator that wires all engine components together and
runs the event-driven loop.

Phase 3 skeleton: the full event loop is wired (feed → event bus → strategy
→ order flush), but portfolio tracking, order execution, slippage, commission,
and analytics are not yet integrated (Phase 4 / Phase 6).
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from backtester.core.event_bus import SimpleEventBus
from backtester.exceptions import BacktestError, ConfigurationError, StrategyError
from backtester.interfaces import (
    BacktestResult,
    Bar,
    CommissionModel,
    DataFeed,
    Event,
    EventType,
    ExecutionModel,
    Order,
    Reporter,
    RiskModel,
    SlippageModel,
    Strategy,
)

logger = logging.getLogger(__name__)


class Backtester:
    """
    Orchestrates a complete backtest run.

    Wires together a ``DataFeed``, ``Strategy``, ``ExecutionModel``,
    ``SlippageModel``, ``CommissionModel``, ``RiskModel``, and ``Reporter``
    into a single run managed by an internal ``SimpleEventBus``.

    Phase 3 status
    --------------
    The event loop iterates the feed, emits ``BarEvent`` events, and calls
    ``Strategy.on_bar`` at the correct point in the bar lifecycle.  Pending
    orders are collected per bar and stored for Phase 4 processing.
    Portfolio tracking, execution, slippage, commission, and tearsheet
    generation are *not* yet active.

    Parameters
    ----------
    feed:       Source of OHLCV bars.
    strategy:   Strategy to run.
    execution:  Order execution model (wired in Phase 4).
    slippage:   Slippage model (wired in Phase 4).
    commission: Commission model (wired in Phase 4).
    risk:       Risk model (wired in Phase 4).
    reporter:   Output reporter (wired in Phase 6).
    debug:      When ``True``, log every bar, emitted event, and flushed
                order at ``DEBUG`` level using Python's ``logging`` module.

    Raises
    ------
    ConfigurationError
        If any required component is ``None`` or is not an instance of the
        expected interface.
    """

    def __init__(
        self,
        feed: DataFeed,
        strategy: Strategy,
        execution: ExecutionModel,
        slippage: SlippageModel,
        commission: CommissionModel,
        risk: RiskModel,
        reporter: Reporter,
        debug: bool = False,
    ) -> None:
        self._validate_components(
            feed=feed,
            strategy=strategy,
            execution=execution,
            slippage=slippage,
            commission=commission,
            risk=risk,
            reporter=reporter,
        )

        self._feed = feed
        self._strategy = strategy
        self._execution = execution
        self._slippage = slippage
        self._commission = commission
        self._risk = risk
        self._reporter = reporter
        self._debug = debug

        # Internal event bus — instantiated here, not injected, so the
        # Backtester controls the full lifecycle.
        self._bus = SimpleEventBus()

        # Accumulated pending orders (execution in Phase 4).
        self._pending_orders: list[Order] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> BacktestResult:
        """
        Execute the backtest and return a ``BacktestResult``.

        Event-driven loop order per bar (Phase 3 skeleton)
        ---------------------------------------------------
        1.  Set ``strategy._current_bar`` to the incoming bar.
        2.  Emit a ``BarEvent`` on the internal ``EventBus``.
        3.  Call ``strategy.on_bar(bar)`` if the strategy's warmup period
            has elapsed (governed by ``strategy._should_process_bar``).
        4.  Flush pending orders from the strategy and accumulate them
            (execution will be wired in Phase 4).

        After all bars: call ``strategy.on_end()`` and return a placeholder
        ``BacktestResult`` with empty equity curve, trades, and metrics.

        Returns
        -------
        BacktestResult
            Phase 3 returns a skeleton result; full metrics are populated
            in Phase 6.
        """
        strategy = self._strategy

        if self._debug:
            logger.debug(
                "Backtester starting — strategy=%s feed=%s",
                type(strategy).__name__,
                type(self._feed).__name__,
            )

        # Inject runtime context into the strategy before on_start.
        strategy._set_context(
            feed=self._feed,
            portfolio=None,  # Wired in Phase 4
            event_bus=self._bus,
        )

        try:
            strategy.on_start(None)
        except Exception as exc:
            raise StrategyError(
                f"strategy.on_start() raised an error: {exc}",
                strategy_name=type(strategy).__name__,
            ) from exc

        bar_index: int = 0
        self._pending_orders.clear()

        for bar in self._feed:
            bar_index += 1

            if self._debug:
                logger.debug(
                    "Bar %d | %s | O=%.4f H=%.4f L=%.4f C=%.4f V=%.0f",
                    bar_index,
                    bar.symbol,
                    bar.open,
                    bar.high,
                    bar.low,
                    bar.close,
                    bar.volume,
                )

            # Make bar available via strategy.current_bar before any dispatch.
            strategy._update_current_bar(bar)

            # Step 1 — emit BarEvent (RiskModel + ExecutionModel + Portfolio
            #           will subscribe in Phase 4).
            bar_event = Event(
                type=EventType.BAR,
                payload=bar,
                timestamp=bar.timestamp,
            )
            if self._debug:
                logger.debug("Emitting %s", bar_event.type.name)
            self._bus.emit(bar_event)

            # Step 2 — call on_bar if warmup period has elapsed.
            if strategy._should_process_bar(bar_index):
                try:
                    strategy.on_bar(bar, None)
                except NotImplementedError:
                    raise
                except Exception as exc:
                    raise StrategyError(
                        f"strategy.on_bar() raised an error at bar {bar_index}: {exc}",
                        strategy_name=type(strategy).__name__,
                        symbol=bar.symbol,
                        bar_index=bar_index,
                    ) from exc

            # Step 3 — collect pending orders (execution in Phase 4).
            new_orders = strategy._flush_orders()
            new_cancels = strategy._flush_cancellations()
            self._pending_orders.extend(new_orders)

            if self._debug and (new_orders or new_cancels):
                logger.debug(
                    "Bar %d | flushed %d order(s), %d cancellation(s)",
                    bar_index,
                    len(new_orders),
                    len(new_cancels),
                )

        try:
            strategy.on_end(None)
        except Exception as exc:
            raise StrategyError(
                f"strategy.on_end() raised an error: {exc}",
                strategy_name=type(strategy).__name__,
            ) from exc

        if self._debug:
            logger.debug(
                "Backtester finished — %d bars, %d pending orders",
                bar_index,
                len(self._pending_orders),
            )

        return BacktestResult(
            equity_curve=pd.Series(dtype=float),
            trades=pd.DataFrame(),
            metrics={},
            params=strategy.params,
            strategy_name=type(strategy).__name__,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_components(
        feed: Any,
        strategy: Any,
        execution: Any,
        slippage: Any,
        commission: Any,
        risk: Any,
        reporter: Any,
    ) -> None:
        """
        Raise ``ConfigurationError`` if any component is missing or has the
        wrong type.
        """
        required: list[tuple[str, Any, type]] = [
            ("feed",       feed,       DataFeed),
            ("strategy",   strategy,   Strategy),
            ("execution",  execution,  ExecutionModel),
            ("slippage",   slippage,   SlippageModel),
            ("commission", commission, CommissionModel),
            ("risk",       risk,       RiskModel),
            ("reporter",   reporter,   Reporter),
        ]
        for name, component, expected_type in required:
            if component is None:
                raise ConfigurationError(
                    f"Required component '{name}' is None.  "
                    f"Provide a {expected_type.__name__} instance."
                )
            if not isinstance(component, expected_type):
                raise ConfigurationError(
                    f"Component '{name}' must be an instance of "
                    f"{expected_type.__name__}, got "
                    f"{type(component).__name__}."
                )
