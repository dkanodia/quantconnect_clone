"""
Typed custom exceptions for the backtesting engine.

Every exception carries optional context fields so that upstream error handlers
(debug mode, test assertions, UI) can display rich diagnostic information
without having to parse message strings.

Hierarchy
---------
BacktestError                           ← root
  DataFeedError
    DataNotFoundError
    DataCorruptionError
  StrategyError
    InvalidSignalError
  OrderError                            ← includes order_id context field
    InvalidOrderError
    OrderRejectedError                  ← includes reason field
    ExecutionError
    InsufficientFundsError              ← includes required / available
    InsufficientPositionError           ← includes requested / held
  PortfolioError
    NegativeCashError
  RiskModelError
    RiskRejectionError                  ← NEW: soft rejection, caught and continued
      PositionSizeLimitExceeded         ← includes requested_pct / limit_pct
    MaxDrawdownExceeded                 ← hard halt: propagates through Backtester
  OptimizationError
    WFOError
  ConfigurationError
"""

from __future__ import annotations

from typing import Any, Optional


class BacktestError(Exception):
    """
    Root exception for all backtesting errors.

    All custom exceptions in this engine inherit from ``BacktestError`` so
    callers can catch the entire family with a single except clause.

    Parameters
    ----------
    message:            Human-readable description of the error.
    strategy_name:      Name of the strategy that triggered the error.
    symbol:             Symbol being processed when the error occurred.
    bar_index:          1-based index of the bar being processed.
    portfolio_snapshot: JSON-serializable dict snapshot of portfolio state at
                        error time.  Keys: ``cash``, ``total_equity``,
                        ``positions``.
    """

    def __init__(
        self,
        message: str,
        *,
        strategy_name: Optional[str] = None,
        symbol: Optional[str] = None,
        bar_index: Optional[int] = None,
        portfolio_snapshot: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.strategy_name = strategy_name
        self.symbol = symbol
        self.bar_index = bar_index
        self.portfolio_snapshot = portfolio_snapshot

    def __str__(self) -> str:
        parts = [super().__str__()]
        if self.strategy_name:
            parts.append(f"strategy={self.strategy_name!r}")
        if self.symbol:
            parts.append(f"symbol={self.symbol!r}")
        if self.bar_index is not None:
            parts.append(f"bar_index={self.bar_index}")
        if self.portfolio_snapshot is not None:
            parts.append("portfolio_snapshot=<captured>")
        return " | ".join(parts)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({str(self)!r})"


# ---------------------------------------------------------------------------
# Data layer errors
# ---------------------------------------------------------------------------


class DataFeedError(BacktestError):
    """Raised when a DataFeed cannot load or iterate data."""


class DataNotFoundError(DataFeedError):
    """Raised when the requested symbol or date range has no data."""


class DataCorruptionError(DataFeedError):
    """Raised when data is malformed (bad schema, NaN prices, etc.)."""


# ---------------------------------------------------------------------------
# Strategy errors
# ---------------------------------------------------------------------------


class StrategyError(BacktestError):
    """Raised when a strategy implementation raises an unexpected exception."""


class InvalidSignalError(StrategyError):
    """Raised when a vectorized strategy returns a signal Series with bad shape or dtype."""


# ---------------------------------------------------------------------------
# Order and execution errors
# ---------------------------------------------------------------------------


class OrderError(BacktestError):
    """
    Raised for order-level problems (invalid quantity, bad price, etc.).

    Parameters
    ----------
    order_id: The unique identifier of the offending order, if known.
    """

    def __init__(
        self,
        message: str,
        *,
        order_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.order_id = order_id

    def __str__(self) -> str:
        base = super().__str__()
        if self.order_id:
            return base + f" | order_id={self.order_id!r}"
        return base


class InvalidOrderError(OrderError):
    """Raised when an order is structurally invalid before submission."""


class OrderRejectedError(OrderError):
    """
    Raised when a RiskModel rejects an order.

    Parameters
    ----------
    reason: Human-readable description of why the order was rejected.
    """

    def __init__(
        self,
        message: str,
        *,
        reason: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)  # order_id forwarded via **kwargs
        self.reason = reason

    def __str__(self) -> str:
        base = super().__str__()
        if self.reason:
            return base + f" | reason={self.reason!r}"
        return base


class ExecutionError(BacktestError):
    """Raised when the ExecutionModel cannot produce a valid fill."""


class InsufficientFundsError(OrderError):
    """Raised when a BUY order exceeds available cash."""

    def __init__(
        self,
        message: str,
        *,
        required: Optional[float] = None,
        available: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.required = required
        self.available = available

    def __str__(self) -> str:
        base = super().__str__()
        extra_parts = []
        if self.required is not None:
            extra_parts.append(f"required={self.required:.2f}")
        if self.available is not None:
            extra_parts.append(f"available={self.available:.2f}")
        if extra_parts:
            return base + " | " + " | ".join(extra_parts)
        return base


class InsufficientPositionError(OrderError):
    """Raised when a SELL order exceeds the current held position."""

    def __init__(
        self,
        message: str,
        *,
        requested: Optional[float] = None,
        held: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.requested = requested
        self.held = held

    def __str__(self) -> str:
        base = super().__str__()
        extra_parts = []
        if self.requested is not None:
            extra_parts.append(f"requested={self.requested:.4f}")
        if self.held is not None:
            extra_parts.append(f"held={self.held:.4f}")
        if extra_parts:
            return base + " | " + " | ".join(extra_parts)
        return base


# ---------------------------------------------------------------------------
# Portfolio errors
# ---------------------------------------------------------------------------


class PortfolioError(BacktestError):
    """Raised when portfolio state is inconsistent or an operation fails."""


class NegativeCashError(PortfolioError):
    """Raised when a fill would drive cash below zero and the model disallows it."""


# ---------------------------------------------------------------------------
# Risk model errors
# ---------------------------------------------------------------------------


class RiskModelError(BacktestError):
    """Raised when a RiskModel encounters an unexpected internal error."""


class RiskRejectionError(RiskModelError):
    """
    Raised by a RiskModel to perform a *soft* rejection of an order.

    The Backtester catches ``RiskRejectionError``, marks the order
    ``CANCELLED``, logs the rejection, and continues with the next order.
    The backtest run is **not** halted.

    Contrast with ``MaxDrawdownExceeded``, which is a *hard* halt that
    propagates through the Backtester and terminates the run.
    """


class MaxDrawdownExceeded(RiskModelError):
    """
    Raised (hard halt) when portfolio drawdown exceeds the configured limit.

    This is a *hard* halt — it is not a subclass of ``RiskRejectionError``
    and therefore propagates through the Backtester, terminating the run.

    Parameters
    ----------
    current_drawdown: Current drawdown as a fraction (e.g. 0.25 = 25 %).
    limit:            The configured drawdown limit.
    """

    def __init__(
        self,
        message: str,
        *,
        current_drawdown: Optional[float] = None,
        limit: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.current_drawdown = current_drawdown
        self.limit = limit

    def __str__(self) -> str:
        base = super().__str__()
        extra_parts = []
        if self.current_drawdown is not None:
            extra_parts.append(f"drawdown={self.current_drawdown:.2%}")
        if self.limit is not None:
            extra_parts.append(f"limit={self.limit:.2%}")
        if extra_parts:
            return base + " | " + " | ".join(extra_parts)
        return base


class PositionSizeLimitExceeded(RiskRejectionError):
    """
    Raised when an order would exceed the maximum allowed position size.

    This is a *soft* rejection (subclass of ``RiskRejectionError``).  The
    Backtester catches it, cancels the order, and continues.

    Parameters
    ----------
    requested_pct: Estimated order size as a fraction of total equity.
    limit_pct:     The configured maximum position size fraction.
    """

    def __init__(
        self,
        message: str,
        *,
        requested_pct: Optional[float] = None,
        limit_pct: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.requested_pct = requested_pct
        self.limit_pct = limit_pct

    def __str__(self) -> str:
        base = super().__str__()
        extra_parts = []
        if self.requested_pct is not None:
            extra_parts.append(f"requested={self.requested_pct:.2%}")
        if self.limit_pct is not None:
            extra_parts.append(f"limit={self.limit_pct:.2%}")
        if extra_parts:
            return base + " | " + " | ".join(extra_parts)
        return base


# ---------------------------------------------------------------------------
# Optimizer errors
# ---------------------------------------------------------------------------


class OptimizationError(BacktestError):
    """Raised when the optimizer encounters an unrecoverable error."""


class WFOError(OptimizationError):
    """Raised when the walk-forward orchestrator cannot partition the data."""


# ---------------------------------------------------------------------------
# Configuration errors
# ---------------------------------------------------------------------------


class ConfigurationError(BacktestError):
    """Raised when the Backtester is constructed with incompatible components."""
