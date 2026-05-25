"""
Risk model implementations.

All models implement RiskModel.check(order, portfolio, bar) → bool and
RiskModel.on_fill(fill, portfolio) → None.

Exception propagation contract
-------------------------------
``PositionSizeLimitExceeded`` (a ``RiskRejectionError``) — caught by the
Backtester, order cancelled, run continues.

``MaxDrawdownExceeded`` (a plain ``RiskModelError``) — propagates through
the Backtester and halts the run.

Available models
----------------
NoRisk               — always approves every order (useful for testing).
MaxDrawdownHalt      — raises MaxDrawdownExceeded and halts the backtest when
                       portfolio drawdown from its peak exceeds a limit.
PositionSizeLimit    — raises PositionSizeLimitExceeded (soft rejection) when
                       a single order's notional value would exceed a configured
                       fraction of total portfolio equity.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from backtester.exceptions import MaxDrawdownExceeded, PositionSizeLimitExceeded
from backtester.interfaces import Bar, Fill, Order, PortfolioTracker, RiskModel

logger = logging.getLogger(__name__)


def _portfolio_snapshot(portfolio: PortfolioTracker) -> Optional[dict[str, Any]]:
    """
    Build a JSON-serialisable portfolio snapshot from any PortfolioTracker.

    Uses only the public ABC methods so it works with both real portfolios
    and test mocks.  Returns ``None`` if the portfolio does not expose the
    expected interface (e.g. a minimal stub in tests).
    """
    # Prefer the richer snapshot() method if available (SimplePortfolio).
    if hasattr(portfolio, "snapshot") and callable(portfolio.snapshot):
        try:
            return portfolio.snapshot()  # type: ignore[union-attr]
        except Exception:
            pass

    # Fall back to building the snapshot from ABC methods.
    try:
        positions: dict[str, dict[str, float]] = {}
        raw = portfolio.get_positions()
        if isinstance(raw, dict):
            for sym, pos in raw.items():
                positions[sym] = {
                    "quantity": float(getattr(pos, "quantity", 0)),
                    "avg_entry_price": float(getattr(pos, "avg_entry_price", 0)),
                    "unrealized_pnl": float(getattr(pos, "unrealized_pnl", 0)),
                    "realized_pnl": float(getattr(pos, "realized_pnl", 0)),
                }
        return {
            "cash": float(portfolio.get_cash()),
            "total_equity": float(portfolio.get_total_equity()),
            "positions": positions,
        }
    except Exception:
        return None


class NoRisk(RiskModel):
    """Approves every order unconditionally.  Useful for isolated testing."""

    def check(self, order: Order, portfolio: PortfolioTracker, current_bar: Bar) -> bool:
        """Always return True."""
        return True

    def on_fill(self, fill: Fill, portfolio: PortfolioTracker) -> None:
        """No-op."""


class MaxDrawdownHalt(RiskModel):
    """
    Halts the backtest when portfolio equity drawdown from its peak exceeds
    the configured limit.

    Parameters
    ----------
    limit : float
        Maximum acceptable drawdown as a fraction (e.g. 0.20 = 20 %).

    Raises
    ------
    MaxDrawdownExceeded
        Raised inside ``check()`` when current drawdown > ``limit``.
        Because this is not a ``RiskRejectionError``, it propagates through
        the Backtester and **halts the run**.
    """

    def __init__(self, limit: float) -> None:
        if not 0.0 < limit <= 1.0:
            raise ValueError(f"MaxDrawdownHalt limit must be in (0, 1], got {limit}.")
        self.limit = limit
        self._peak_equity: Optional[float] = None

    def check(self, order: Order, portfolio: PortfolioTracker, current_bar: Bar) -> bool:
        """Return True if drawdown ≤ limit, otherwise raise MaxDrawdownExceeded."""
        equity = portfolio.get_total_equity()
        if self._peak_equity is None:
            self._peak_equity = equity
        else:
            self._peak_equity = max(self._peak_equity, equity)

        if self._peak_equity > 0:
            drawdown = (self._peak_equity - equity) / self._peak_equity
            if drawdown > self.limit:
                raise MaxDrawdownExceeded(
                    f"Portfolio drawdown {drawdown:.2%} exceeds halt limit {self.limit:.2%}.",
                    current_drawdown=drawdown,
                    limit=self.limit,
                    symbol=order.symbol,
                    portfolio_snapshot=_portfolio_snapshot(portfolio),
                )
        return True

    def on_fill(self, fill: Fill, portfolio: PortfolioTracker) -> None:
        """Update the tracked peak equity after each fill."""
        equity = portfolio.get_total_equity()
        if self._peak_equity is None:
            self._peak_equity = equity
        else:
            self._peak_equity = max(self._peak_equity, equity)


class PositionSizeLimit(RiskModel):
    """
    Soft-rejects orders whose estimated notional value exceeds a fraction of
    total portfolio equity.

    Estimated notional = order.quantity × current_bar.close.

    Parameters
    ----------
    limit_pct : float
        Maximum position size as a fraction of total equity
        (e.g. 0.10 = 10 %).

    Raises
    ------
    PositionSizeLimitExceeded
        A ``RiskRejectionError`` — caught by the Backtester, order cancelled,
        run continues.
    """

    def __init__(self, limit_pct: float) -> None:
        if not 0.0 < limit_pct <= 1.0:
            raise ValueError(
                f"PositionSizeLimit limit_pct must be in (0, 1], got {limit_pct}."
            )
        self.limit_pct = limit_pct

    def check(self, order: Order, portfolio: PortfolioTracker, current_bar: Bar) -> bool:
        """
        Return True if within limit, raise PositionSizeLimitExceeded otherwise.
        """
        total_equity = portfolio.get_total_equity()
        if total_equity <= 0:
            return True

        order_value = order.quantity * current_bar.close
        order_pct = order_value / total_equity

        if order_pct > self.limit_pct:
            raise PositionSizeLimitExceeded(
                f"Order notional {order_pct:.2%} of equity exceeds limit {self.limit_pct:.2%}.",
                requested_pct=order_pct,
                limit_pct=self.limit_pct,
                symbol=order.symbol,
                portfolio_snapshot=_portfolio_snapshot(portfolio),
            )
        return True

    def on_fill(self, fill: Fill, portfolio: PortfolioTracker) -> None:
        """No-op."""
