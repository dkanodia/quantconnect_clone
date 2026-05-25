"""
Backtester — orchestrator that wires all engine components together and
runs the event-driven loop.

Phase 5: integrated DebugLogger, full exception context enrichment, and
enforced exception propagation policy.

Per-bar loop order (8 steps)
-----------------------------
1.  Risk-check all pending orders:
      - ``RiskRejectionError`` → order CANCELLED, logged, loop continues.
      - Other ``BacktestError`` (e.g. ``MaxDrawdownExceeded``) → enriched,
        re-raised (halts run).
2.  Execute approved orders via the ExecutionModel.
3.  For each fill: record in OrderManager → apply to portfolio →
    notify RiskModel → emit FILL event → call strategy.on_order.
4.  Update market prices in the portfolio (``update_market``).
5.  Record current equity (``record_equity``).
6.  Emit a BAR event on the internal EventBus.
7.  Call ``strategy.on_bar`` if the warmup period has elapsed.
8.  Flush new orders/cancellations from the strategy and submit them
    to the OrderManager for execution on the next bar.

Exception propagation policy
------------------------------
| Exception kind               | Action                                    |
|------------------------------|-------------------------------------------|
| User errors                  | Propagate immediately                     |
| Strategy hook errors         | Wrapped in StrategyError with context     |
| RiskRejectionError           | Order cancelled, run continues            |
| Hard risk halts              | Enriched + re-raised (run halts)          |
| PortfolioError               | Propagate immediately                     |
| Unexpected exceptions        | Wrapped in BacktestError with context     |

``current_bar`` / ``next_bar`` semantics for ExecutionModel
------------------------------------------------------------
When processing bars[i]:
  current_bar = bars[i-1]  (bar on which the order was placed)
  next_bar    = bars[i]    (bar now being processed)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

from backtester.core.debug_logger import DebugLogger
from backtester.core.event_bus import SimpleEventBus
from backtester.core.order_manager import OrderManager
from backtester.exceptions import (
    BacktestError,
    ConfigurationError,
    OrderError,
    PortfolioError,
    RiskRejectionError,
    StrategyError,
)
from backtester.interfaces import (
    BacktestResult,
    Bar,
    CommissionModel,
    DataFeed,
    Event,
    EventType,
    ExecutionModel,
    Order,
    OrderStatus,
    Reporter,
    RiskModel,
    SlippageModel,
    Strategy,
)
from backtester.portfolio.simple_portfolio import SimplePortfolio

logger = logging.getLogger(__name__)


class Backtester:
    """
    Orchestrates a complete backtest run.

    Wires together a ``DataFeed``, ``Strategy``, ``ExecutionModel``,
    ``SlippageModel``, ``CommissionModel``, ``RiskModel``, and ``Reporter``
    into a single run managed by an internal ``SimpleEventBus`` and
    ``OrderManager``.

    Parameters
    ----------
    feed:         Source of OHLCV bars.
    strategy:     Strategy to run.
    execution:    Order execution model.
    slippage:     Slippage model (injected into the execution model).
    commission:   Commission model (injected into the execution model).
    risk:         Risk model gate applied before each execution attempt.
    reporter:     Output reporter (called after the run — wired in Phase 6).
    initial_cash: Starting portfolio cash.  Defaults to 100 000.
    debug:        When ``True``, emit structured JSON debug logs via
                  ``DebugLogger``.

    Raises
    ------
    ConfigurationError
        If any required component is None or has the wrong type.
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
        initial_cash: float = 100_000.0,
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
        self._initial_cash = initial_cash
        self._debug = debug

        self._bus = SimpleEventBus()
        self._portfolio = SimplePortfolio(initial_cash)
        self._order_manager = OrderManager(strategy_name=type(strategy).__name__)
        self._debug_logger = DebugLogger(
            strategy_name=type(strategy).__name__,
            enabled=debug,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> BacktestResult:
        """
        Execute the backtest and return a populated ``BacktestResult``.

        The full data set is materialised into memory at the start
        (``bars = list(self._feed)``) to support lookahead for execution
        models.

        Returns
        -------
        BacktestResult
            equity_curve, trades DataFrame, and at minimum these metrics:
            total_return, final_equity, num_trades.

        Raises
        ------
        StrategyError
            If any strategy lifecycle hook (on_start, on_bar, on_end,
            on_order) raises an unexpected exception.
        MaxDrawdownExceeded
            If the MaxDrawdownHalt risk model triggers.
        BacktestError
            If any other unexpected exception occurs in the event loop.
        """
        strategy = self._strategy
        portfolio = self._portfolio
        order_manager = self._order_manager
        bus = self._bus
        dl = self._debug_logger
        strategy_name = type(strategy).__name__

        dl.log_phase("on_start")

        strategy._set_context(
            feed=self._feed,
            portfolio=portfolio,
            event_bus=bus,
        )

        try:
            strategy.on_start(None)
        except Exception as exc:
            raise StrategyError(
                f"strategy.on_start() raised an error: {exc}",
                strategy_name=strategy_name,
            ) from exc

        bars = list(self._feed)
        bar_index = 0
        prev_bar: Optional[Bar] = None

        for bar in bars:
            bar_index += 1
            strategy._update_current_bar(bar)
            dl.log_bar(bar, bar_index)

            # ----------------------------------------------------------
            # Steps 1–3: execute pending orders from the previous bar
            # ----------------------------------------------------------
            if prev_bar is not None:
                self._execute_pending_orders(
                    order_manager=order_manager,
                    portfolio=portfolio,
                    strategy=strategy,
                    bus=bus,
                    prev_bar=prev_bar,
                    current_bar=bar,
                    bar_index=bar_index,
                )

            # ----------------------------------------------------------
            # Step 4: mark-to-market all open positions
            # ----------------------------------------------------------
            portfolio.update_market([bar])

            # ----------------------------------------------------------
            # Step 5: snapshot equity
            # ----------------------------------------------------------
            portfolio.record_equity(bar.timestamp)
            dl.log_equity(bar.timestamp, portfolio.get_total_equity())

            # ----------------------------------------------------------
            # Step 6: emit BAR event
            # ----------------------------------------------------------
            bar_event = Event(type=EventType.BAR, payload=bar, timestamp=bar.timestamp)
            bus.emit(bar_event)
            dl.log_event(bar_event)

            # ----------------------------------------------------------
            # Step 7: call on_bar if warmup period has elapsed
            # ----------------------------------------------------------
            if strategy._should_process_bar(bar_index):
                try:
                    strategy.on_bar(bar, None)
                except NotImplementedError:
                    raise
                except Exception as exc:
                    raise StrategyError(
                        f"strategy.on_bar() raised an error at bar {bar_index}: {exc}",
                        strategy_name=strategy_name,
                        symbol=bar.symbol,
                        bar_index=bar_index,
                    ) from exc

            # ----------------------------------------------------------
            # Step 8: flush and submit new orders / cancellations
            # ----------------------------------------------------------
            new_orders = strategy._flush_orders()
            new_cancels = strategy._flush_cancellations()

            for order in new_orders:
                order_manager.submit(order)
                dl.log_order_submitted(order)

            for order_id in new_cancels:
                try:
                    cancelled_order = order_manager.get_order(order_id)
                    order_manager.cancel(order_id)
                    dl.log_order_cancelled(cancelled_order, "strategy requested cancellation")
                except OrderError as exc:
                    logger.warning("Cancel failed for order %s: %s", order_id, exc)

            prev_bar = bar

        # --------------------------------------------------------------
        # End of data
        # --------------------------------------------------------------
        dl.log_phase("on_end")

        try:
            strategy.on_end(None)
        except Exception as exc:
            raise StrategyError(
                f"strategy.on_end() raised an error: {exc}",
                strategy_name=strategy_name,
            ) from exc

        return self._build_result(strategy, portfolio, order_manager)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _execute_pending_orders(
        self,
        order_manager: OrderManager,
        portfolio: SimplePortfolio,
        strategy: Strategy,
        bus: SimpleEventBus,
        *,
        prev_bar: Bar,
        current_bar: Bar,
        bar_index: int,
    ) -> None:
        """
        Steps 1–3 of the per-bar loop.

        For each pending order:

        1. **Risk check** — ``RiskRejectionError`` → cancel & continue;
           other ``BacktestError`` → enrich & re-raise.
        2. **Execute** via the ``ExecutionModel`` — may return ``None``.
        3. **On fill**: record → apply to portfolio → notify risk →
           emit FILL event → call ``strategy.on_order``.
        """
        strategy_name = type(strategy).__name__
        dl = self._debug_logger

        for order in list(order_manager.get_pending()):

            # ── Step 1: risk gate ──────────────────────────────────────
            try:
                approved = self._risk.check(order, portfolio, current_bar)
            except RiskRejectionError as exc:
                # Soft rejection — cancel order, log, continue.
                self._enrich(exc, current_bar, bar_index)
                order.status = OrderStatus.CANCELLED
                reason = str(exc)
                dl.log_risk_check(order, approved=False, reason=reason)
                dl.log_order_rejected(order, reason)
                logger.info(
                    "Order %s soft-rejected (bar %d): %s", order.id, bar_index, exc
                )
                continue
            except BacktestError as exc:
                # Hard halt (MaxDrawdownExceeded, etc.) — enrich + re-raise.
                self._enrich(exc, current_bar, bar_index)
                raise
            except Exception as exc:
                wrapped = BacktestError(
                    f"Risk model raised an unexpected error: {exc}",
                    symbol=current_bar.symbol,
                    bar_index=bar_index,
                    strategy_name=strategy_name,
                )
                raise wrapped from exc

            dl.log_risk_check(order, approved=True)

            if not approved:
                order.status = OrderStatus.REJECTED
                reason = "risk model returned False"
                dl.log_order_rejected(order, reason)
                continue

            # ── Step 2: execution ──────────────────────────────────────
            fill = self._execution.execute(order, prev_bar, current_bar)
            if fill is None:
                continue  # not fillable yet (e.g. NextOpen at end of data)

            # ── Step 3: record, portfolio, risk feedback, events ───────
            order_manager.record_fill(fill)

            try:
                portfolio.apply_fill(fill)
            except PortfolioError as exc:
                self._enrich(exc, current_bar, bar_index)
                raise

            self._risk.on_fill(fill, portfolio)

            fill_event = Event(
                type=EventType.FILL, payload=fill, timestamp=fill.timestamp
            )
            dl.log_fill(fill)
            bus.emit(fill_event)
            dl.log_event(fill_event)

            try:
                strategy.on_order(order, None)
            except Exception as exc:
                raise StrategyError(
                    f"strategy.on_order() raised an error: {exc}",
                    strategy_name=strategy_name,
                    symbol=order.symbol,
                    bar_index=bar_index,
                ) from exc

    def _enrich(
        self,
        exc: BacktestError,
        bar: Optional[Bar],
        bar_index: int,
    ) -> BacktestError:
        """
        Backfill ``strategy_name``, ``symbol``, and ``bar_index`` into an
        existing ``BacktestError`` **only if those fields are currently None**.

        Existing values are never overwritten so the originating raise site
        retains authority.

        Parameters
        ----------
        exc:       The exception to enrich (mutated in-place).
        bar:       Current bar, used to extract the symbol.
        bar_index: Current 1-based bar index.

        Returns
        -------
        BacktestError
            The same exception object, enriched.
        """
        if exc.strategy_name is None:
            exc.strategy_name = type(self._strategy).__name__
        if exc.symbol is None and bar is not None:
            exc.symbol = bar.symbol
        if exc.bar_index is None:
            exc.bar_index = bar_index
        return exc

    def _build_result(
        self,
        strategy: Strategy,
        portfolio: SimplePortfolio,
        order_manager: OrderManager,
    ) -> BacktestResult:
        """Compute and return the final BacktestResult."""
        equity_curve = portfolio.get_equity_curve()
        trades = portfolio.get_trades()

        final_equity = (
            float(equity_curve.iloc[-1]) if len(equity_curve) > 0 else self._initial_cash
        )
        total_return = (
            (final_equity - self._initial_cash) / self._initial_cash
            if self._initial_cash > 0
            else 0.0
        )

        metrics: dict[str, Any] = {
            "total_return": total_return,
            "final_equity": final_equity,
            "num_trades": len(trades),
        }

        return BacktestResult(
            equity_curve=equity_curve,
            trades=trades,
            metrics=metrics,
            params=strategy.params,
            strategy_name=type(strategy).__name__,
        )

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
        """Raise ConfigurationError if any component is missing or wrong type."""
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
                    f"{expected_type.__name__}, got {type(component).__name__}."
                )
