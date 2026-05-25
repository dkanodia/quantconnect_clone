"""
Slippage model implementations.

All models implement SlippageModel.adjust(raw_price, order, bar) → float.

Available models
----------------
ZeroSlippage      — no adjustment (useful for testing).
FixedSlippage     — fixed absolute amount added/subtracted per trade.
PercentSlippage   — percentage of raw price added/subtracted.
VolumeSlippage    — impact scales with order size relative to bar volume,
                    clamped at 5× the base percentage.
"""

from __future__ import annotations

from backtester.interfaces import Bar, Order, OrderSide, SlippageModel


class ZeroSlippage(SlippageModel):
    """No slippage — raw fill price is returned unchanged."""

    def adjust(self, raw_price: float, order: Order, bar: Bar) -> float:
        return raw_price


class FixedSlippage(SlippageModel):
    """
    Constant slippage amount added (BUY) or subtracted (SELL).

    Parameters
    ----------
    amount : float
        Cash amount of slippage per share / unit.
    """

    def __init__(self, amount: float) -> None:
        if amount < 0:
            raise ValueError(f"FixedSlippage amount must be >= 0, got {amount}.")
        self.amount = amount

    def adjust(self, raw_price: float, order: Order, bar: Bar) -> float:
        if order.side == OrderSide.BUY:
            return raw_price + self.amount
        return raw_price - self.amount


class PercentSlippage(SlippageModel):
    """
    Slippage as a fraction of the raw fill price.

    Parameters
    ----------
    pct : float
        Fraction of price (e.g. 0.001 = 0.1 %).  Must be >= 0.
    """

    def __init__(self, pct: float) -> None:
        if pct < 0:
            raise ValueError(f"PercentSlippage pct must be >= 0, got {pct}.")
        self.pct = pct

    def adjust(self, raw_price: float, order: Order, bar: Bar) -> float:
        if order.side == OrderSide.BUY:
            return raw_price * (1.0 + self.pct)
        return raw_price * (1.0 - self.pct)


class VolumeSlippage(SlippageModel):
    """
    Market-impact model: slippage scales with order size relative to bar volume.

    impact = pct × (quantity / volume),  clamped at 5 × pct.

    Parameters
    ----------
    pct : float
        Base slippage fraction.  Must be >= 0.
    """

    def __init__(self, pct: float) -> None:
        if pct < 0:
            raise ValueError(f"VolumeSlippage pct must be >= 0, got {pct}.")
        self.pct = pct

    def adjust(self, raw_price: float, order: Order, bar: Bar) -> float:
        volume = bar.volume if bar.volume > 0 else 1.0
        impact = self.pct * order.quantity / volume
        impact = min(impact, 5.0 * self.pct)
        if order.side == OrderSide.BUY:
            return raw_price * (1.0 + impact)
        return raw_price * (1.0 - impact)
