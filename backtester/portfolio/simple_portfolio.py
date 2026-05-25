"""
SimplePortfolio — a straightforward cash-and-positions portfolio tracker.

Tracks:
- Available cash
- Open positions (quantity, average entry price, unrealised / realised PnL)
- Equity curve (total equity sampled at each bar)
- Completed trade records (every SELL that reduces a position)

All exceptions raised here carry a ``portfolio_snapshot`` so callers can
inspect the full portfolio state at the point of failure.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Sequence

import pandas as pd

from backtester.exceptions import (
    InsufficientFundsError,
    InsufficientPositionError,
    NegativeCashError,
)
from backtester.interfaces import Bar, Fill, OrderSide, Position, PortfolioTracker

logger = logging.getLogger(__name__)

_FLOAT_EPSILON = 1e-9


class SimplePortfolio(PortfolioTracker):
    """
    Single-currency, long-only portfolio tracker.

    Parameters
    ----------
    initial_cash : float
        Starting cash balance.  Defaults to 100 000.

    Notes
    -----
    - Short selling is not supported; a SELL must be backed by an existing
      position.
    - Fractional shares are allowed.
    - Average entry price is updated on every BUY using a quantity-weighted
      mean (average-cost method).
    - Each SELL fill closes or reduces a position and records a trade.
    - All raised exceptions carry a ``portfolio_snapshot`` for diagnostics.
    """

    def __init__(self, initial_cash: float = 100_000.0) -> None:
        self._initial_cash = initial_cash
        self._cash: float = initial_cash
        self._positions: dict[str, Position] = {}
        self._last_prices: dict[str, float] = {}
        self._equity_snapshots: list[tuple[datetime, float]] = []
        self._trade_records: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # PortfolioTracker interface
    # ------------------------------------------------------------------

    def apply_fill(self, fill: Fill) -> None:
        """
        Update cash and positions given a completed fill.

        BUY path
        --------
        cost = price × qty + commission
        Raises ``InsufficientFundsError`` if cost > cash.
        Creates or updates the position (average-cost method).

        SELL path
        ---------
        Raises ``InsufficientPositionError`` if qty > held quantity.
        proceeds = price × qty − commission
        Raises ``NegativeCashError`` if proceeds would drive cash below zero.
        Records realised PnL and a trade entry.

        All exceptions include ``portfolio_snapshot`` for diagnostics.
        """
        if fill.side == OrderSide.BUY:
            self._apply_buy(fill)
        else:
            self._apply_sell(fill)

        self._last_prices[fill.symbol] = fill.price

    def update_market(self, bars: Sequence[Bar]) -> None:
        """Revalue open positions using the close price of each bar."""
        for bar in bars:
            self._last_prices[bar.symbol] = bar.close
            pos = self._positions.get(bar.symbol)
            if pos is not None:
                pos.unrealized_pnl = (bar.close - pos.avg_entry_price) * pos.quantity

    def record_equity(self, timestamp: datetime) -> None:
        """Append the current total equity to the equity curve."""
        self._equity_snapshots.append((timestamp, self.get_total_equity()))

    def get_equity_curve(self) -> pd.Series:
        """Return a datetime-indexed Series of total equity values."""
        if not self._equity_snapshots:
            return pd.Series(dtype=float)
        timestamps, values = zip(*self._equity_snapshots)
        return pd.Series(list(values), index=pd.DatetimeIndex(list(timestamps)), name="equity")

    def get_positions(self) -> dict[str, Position]:
        """Return a snapshot of all open positions keyed by symbol."""
        return dict(self._positions)

    def get_cash(self) -> float:
        """Return current available cash."""
        return self._cash

    def get_total_equity(self) -> float:
        """Return cash plus mark-to-market value of all open positions."""
        position_value = sum(
            pos.quantity * self._last_prices.get(symbol, pos.avg_entry_price)
            for symbol, pos in self._positions.items()
        )
        return self._cash + position_value

    def get_trades(self) -> pd.DataFrame:
        """
        Return a DataFrame of every completed trade (each SELL fill).

        Columns: symbol, exit_timestamp, quantity, entry_price, exit_price,
                 pnl, commission.
        """
        if not self._trade_records:
            return pd.DataFrame(
                columns=[
                    "symbol", "exit_timestamp", "quantity",
                    "entry_price", "exit_price", "pnl", "commission",
                ]
            )
        return pd.DataFrame(self._trade_records)

    # ------------------------------------------------------------------
    # Extra helpers (not in the ABC)
    # ------------------------------------------------------------------

    @property
    def initial_cash(self) -> float:
        """Return the cash balance the portfolio was initialised with."""
        return self._initial_cash

    def snapshot(self) -> dict[str, Any]:
        """
        Return a JSON-serialisable snapshot of the current portfolio state.

        Structure
        ---------
        .. code-block:: python

            {
                "cash": float,
                "total_equity": float,
                "positions": {
                    "<symbol>": {
                        "quantity": float,
                        "avg_entry_price": float,
                        "unrealized_pnl": float,
                        "realized_pnl": float,
                    }
                }
            }

        All raise sites in this class pass ``portfolio_snapshot=self.snapshot()``
        so the full portfolio state is captured at the point of error.
        """
        positions: dict[str, dict[str, float]] = {}
        for symbol, pos in self._positions.items():
            positions[symbol] = {
                "quantity": pos.quantity,
                "avg_entry_price": pos.avg_entry_price,
                "unrealized_pnl": pos.unrealized_pnl,
                "realized_pnl": pos.realized_pnl,
            }
        return {
            "cash": self._cash,
            "total_equity": self.get_total_equity(),
            "positions": positions,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_buy(self, fill: Fill) -> None:
        cost = fill.price * fill.quantity + fill.commission
        if cost > self._cash:
            raise InsufficientFundsError(
                f"Cannot buy {fill.quantity} {fill.symbol} @ {fill.price:.4f}: "
                f"cost {cost:.2f} exceeds available cash {self._cash:.2f}.",
                required=cost,
                available=self._cash,
                symbol=fill.symbol,
                portfolio_snapshot=self.snapshot(),
            )
        self._cash -= cost

        pos = self._positions.get(fill.symbol)
        if pos is None:
            self._positions[fill.symbol] = Position(
                symbol=fill.symbol,
                quantity=fill.quantity,
                avg_entry_price=fill.price,
            )
        else:
            total_qty = pos.quantity + fill.quantity
            pos.avg_entry_price = (
                pos.avg_entry_price * pos.quantity + fill.price * fill.quantity
            ) / total_qty
            pos.quantity = total_qty

        logger.debug(
            "BUY applied: symbol=%s qty=%s price=%.4f commission=%.4f cash_after=%.2f",
            fill.symbol, fill.quantity, fill.price, fill.commission, self._cash,
        )

    def _apply_sell(self, fill: Fill) -> None:
        pos = self._positions.get(fill.symbol)
        held = pos.quantity if pos is not None else 0.0

        if pos is None or fill.quantity > held + _FLOAT_EPSILON:
            raise InsufficientPositionError(
                f"Cannot sell {fill.quantity} {fill.symbol}: only {held:.4f} held.",
                requested=fill.quantity,
                held=held,
                symbol=fill.symbol,
                portfolio_snapshot=self.snapshot(),
            )

        proceeds = fill.price * fill.quantity - fill.commission
        pnl = (fill.price - pos.avg_entry_price) * fill.quantity - fill.commission
        pos.realized_pnl += pnl

        new_cash = self._cash + proceeds
        if new_cash < -_FLOAT_EPSILON:
            raise NegativeCashError(
                f"SELL of {fill.quantity} {fill.symbol} would drive cash to "
                f"{new_cash:.2f}.",
                symbol=fill.symbol,
                portfolio_snapshot=self.snapshot(),
            )
        self._cash = max(new_cash, 0.0)

        self._trade_records.append(
            {
                "symbol": fill.symbol,
                "exit_timestamp": fill.timestamp,
                "quantity": fill.quantity,
                "entry_price": pos.avg_entry_price,
                "exit_price": fill.price,
                "pnl": pnl,
                "commission": fill.commission,
            }
        )

        pos.quantity -= fill.quantity
        if pos.quantity < _FLOAT_EPSILON:
            del self._positions[fill.symbol]

        logger.debug(
            "SELL applied: symbol=%s qty=%s price=%.4f pnl=%.4f cash_after=%.2f",
            fill.symbol, fill.quantity, fill.price, pnl, self._cash,
        )
