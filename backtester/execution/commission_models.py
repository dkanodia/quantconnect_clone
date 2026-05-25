"""
Commission model implementations.

All models implement CommissionModel.compute(fill) → float.

Available models
----------------
ZeroCommission    — no commission (useful for testing).
FixedPerTrade     — flat fee per trade, regardless of size.
PercentCommission — percentage of trade value (price × quantity).
TieredCommission  — rate determined by the highest tier whose min_value
                    threshold the trade value meets.
"""

from __future__ import annotations

from backtester.interfaces import CommissionModel, Fill


class ZeroCommission(CommissionModel):
    """No commission — always returns 0.0."""

    def compute(self, fill: Fill) -> float:
        return 0.0


class FixedPerTrade(CommissionModel):
    """
    Flat commission amount charged on every trade.

    Parameters
    ----------
    amount : float
        Commission in cash per trade.  Must be >= 0.
    """

    def __init__(self, amount: float) -> None:
        if amount < 0:
            raise ValueError(f"FixedPerTrade amount must be >= 0, got {amount}.")
        self.amount = amount

    def compute(self, fill: Fill) -> float:
        return self.amount


class PercentCommission(CommissionModel):
    """
    Commission as a fraction of the trade's notional value.

    commission = price × quantity × pct

    Parameters
    ----------
    pct : float
        Fraction of notional (e.g. 0.001 = 0.1 %).  Must be >= 0.
    """

    def __init__(self, pct: float) -> None:
        if pct < 0:
            raise ValueError(f"PercentCommission pct must be >= 0, got {pct}.")
        self.pct = pct

    def compute(self, fill: Fill) -> float:
        return fill.price * fill.quantity * self.pct


class TieredCommission(CommissionModel):
    """
    Volume-tiered commission schedule.

    The commission rate applied is from the *highest* tier whose
    ``min_value`` is ≤ the trade's notional value.  If no tier qualifies,
    commission is 0.

    Parameters
    ----------
    tiers : list[tuple[float, float]]
        Each tuple is (min_value, commission_rate).  Tiers are sorted
        internally by min_value (ascending), so declaration order does
        not matter.

    Example
    -------
    TieredCommission([(0, 0.002), (10_000, 0.001), (100_000, 0.0005)])
    """

    def __init__(self, tiers: list[tuple[float, float]]) -> None:
        if not tiers:
            raise ValueError("TieredCommission requires at least one tier.")
        self.tiers = sorted(tiers, key=lambda t: t[0])

    def compute(self, fill: Fill) -> float:
        trade_value = fill.price * fill.quantity
        rate = 0.0
        for min_value, commission_rate in self.tiers:
            if trade_value >= min_value:
                rate = commission_rate
        return trade_value * rate
