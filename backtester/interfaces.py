"""
All abstract base classes, dataclasses, and enums for the backtesting engine.

Nothing concrete lives here — only contracts that the rest of the system
must implement. Every other module imports shared types from this file.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Generator, Iterator, Optional, Sequence

import pandas as pd


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(Enum):
    CREATED = "CREATED"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class Visibility(Enum):
    PRIVATE = "PRIVATE"
    TEAM = "TEAM"
    FEATURED = "FEATURED"


class EventType(Enum):
    BAR = auto()
    ORDER = auto()
    FILL = auto()
    PORTFOLIO_UPDATE = auto()
    RISK_CHECK = auto()
    BACKTEST_START = auto()
    BACKTEST_END = auto()


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------


@dataclass
class Bar:
    """A single OHLCV bar for one symbol."""

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Order:
    """A trading order submitted by a strategy."""

    symbol: str
    side: OrderSide
    type: OrderType
    quantity: float
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: OrderStatus = OrderStatus.CREATED

    def __post_init__(self) -> None:
        if self.type in (OrderType.LIMIT, OrderType.STOP_LIMIT) and self.limit_price is None:
            raise ValueError(f"Order type {self.type} requires a limit_price.")
        if self.type in (OrderType.STOP, OrderType.STOP_LIMIT) and self.stop_price is None:
            raise ValueError(f"Order type {self.type} requires a stop_price.")


@dataclass
class Fill:
    """Confirmation that an order was executed."""

    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    commission: float
    timestamp: datetime


@dataclass
class Position:
    """Current holding for a single symbol."""

    symbol: str
    quantity: float
    avg_entry_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    def market_value(self, current_price: float) -> float:
        """Mark-to-market value of this position."""
        return self.quantity * current_price


@dataclass
class BacktestResult:
    """The complete output of a single backtest run."""

    equity_curve: pd.Series
    trades: pd.DataFrame
    metrics: dict[str, Any]
    params: dict[str, Any]
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    owner: str = ""
    visibility: Visibility = Visibility.PRIVATE
    strategy_name: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Event:
    """A typed event dispatched through the EventBus."""

    type: EventType
    payload: Any
    timestamp: datetime = field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Abstract interfaces
# ---------------------------------------------------------------------------


class DataFeed(ABC):
    """Yields Bar objects one at a time for a given symbol / date range."""

    @abstractmethod
    def __iter__(self) -> Iterator[Bar]:
        """Return an iterator that yields Bar objects in chronological order."""

    @abstractmethod
    def reset(self) -> None:
        """Reset the feed to its initial state so it can be iterated again."""

    @property
    @abstractmethod
    def symbols(self) -> list[str]:
        """Return the list of symbols provided by this feed."""


class Strategy(ABC):
    """
    Base contract for all strategies.

    Concrete strategies inherit from either EventDrivenStrategy or
    VectorizedStrategy in backtester/strategy/, not directly from this class.
    """

    @abstractmethod
    def on_start(self, context: Any) -> None:
        """Called once before the first bar. Use for initialisation."""

    @abstractmethod
    def on_bar(self, bar: Bar, context: Any) -> None:
        """Called on every bar in event-driven mode."""

    @abstractmethod
    def on_order(self, order: Order, context: Any) -> None:
        """Called when an order status changes."""

    @abstractmethod
    def on_end(self, context: Any) -> None:
        """Called once after the last bar. Use for cleanup / final signals."""

    @abstractmethod
    def get_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Return a Series of float signals indexed by timestamp (vectorized mode).

        Positive values → long signal, negative → short, zero → flat.
        """


class ExecutionModel(ABC):
    """Determines when and at what price an order is filled."""

    @abstractmethod
    def execute(
        self,
        order: Order,
        current_bar: Bar,
        next_bar: Optional[Bar],
    ) -> Optional[Fill]:
        """
        Given an order and bar context, return a Fill or None if not fillable yet.

        Parameters
        ----------
        order:       The pending order to attempt to fill.
        current_bar: The bar during which the order was submitted.
        next_bar:    The next bar (may be None at end of data).
        """


class SlippageModel(ABC):
    """Adjusts a raw fill price to account for market impact."""

    @abstractmethod
    def adjust(self, raw_price: float, order: Order, bar: Bar) -> float:
        """
        Return the slippage-adjusted fill price.

        Parameters
        ----------
        raw_price: The unadjusted price from the ExecutionModel.
        order:     The order being filled.
        bar:       The bar at which execution occurs.
        """


class CommissionModel(ABC):
    """Computes the commission cost for a fill."""

    @abstractmethod
    def compute(self, fill: Fill) -> float:
        """
        Return commission cost in cash.

        Parameters
        ----------
        fill: The fill (price and quantity already set, commission not yet set).
        """


class PortfolioTracker(ABC):
    """Tracks cash, positions, unrealized/realized PnL, and the equity curve."""

    @abstractmethod
    def apply_fill(self, fill: Fill) -> None:
        """Update portfolio state given an executed fill."""

    @abstractmethod
    def update_market(self, bars: Sequence[Bar]) -> None:
        """Revalue all open positions using the latest bar prices."""

    @abstractmethod
    def record_equity(self, timestamp: datetime) -> None:
        """Append the current total equity to the equity curve."""

    @abstractmethod
    def get_equity_curve(self) -> pd.Series:
        """Return the full equity curve as a datetime-indexed Series."""

    @abstractmethod
    def get_positions(self) -> dict[str, Position]:
        """Return a snapshot of all open positions keyed by symbol."""

    @abstractmethod
    def get_cash(self) -> float:
        """Return current available cash."""

    @abstractmethod
    def get_total_equity(self) -> float:
        """Return total portfolio value (cash + marked-to-market positions)."""

    @abstractmethod
    def get_trades(self) -> pd.DataFrame:
        """Return a DataFrame of all completed round-trip trades."""


class RiskModel(ABC):
    """Decides whether a proposed order is allowed given current portfolio state."""

    @abstractmethod
    def check(
        self,
        order: Order,
        portfolio: PortfolioTracker,
        current_bar: Bar,
    ) -> bool:
        """
        Return True if the order is approved, False if it should be rejected.

        Parameters
        ----------
        order:       The proposed order.
        portfolio:   Current portfolio state.
        current_bar: The bar at which the check is performed.
        """

    @abstractmethod
    def on_fill(self, fill: Fill, portfolio: PortfolioTracker) -> None:
        """Called after every fill so the risk model can update internal state."""


class EventBus(ABC):
    """
    Synchronous publish/subscribe event dispatcher.

    Handlers are called in registration order.
    """

    @abstractmethod
    def subscribe(self, event_type: EventType, handler: Any) -> None:
        """Register a callable to be invoked when event_type is emitted."""

    @abstractmethod
    def unsubscribe(self, event_type: EventType, handler: Any) -> None:
        """Remove a previously registered handler."""

    @abstractmethod
    def emit(self, event: Event) -> None:
        """Dispatch an event synchronously to all registered handlers."""


class Optimizer(ABC):
    """
    Searches a parameter space to maximise an objective function.

    The objective function signature is:
        (strategy_class, params: dict, **kwargs) -> float
    """

    @abstractmethod
    def optimize(
        self,
        strategy_class: type,
        param_space: dict[str, Any],
        objective: Any,
        n_trials: int = 100,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Run the optimisation and return the best parameter dict.

        Parameters
        ----------
        strategy_class: The strategy class to instantiate per trial.
        param_space:    Mapping of param name → search space descriptor.
        objective:      Callable that scores a trial (higher is better).
        n_trials:       Maximum number of trials / grid points.
        **kwargs:       Passed through to the objective function.
        """


class Reporter(ABC):
    """Consumes a completed BacktestResult and produces output."""

    @abstractmethod
    def report(self, result: BacktestResult) -> Any:
        """
        Generate and return the report in whatever format the implementation uses.

        Parameters
        ----------
        result: The completed backtest result to report on.
        """
