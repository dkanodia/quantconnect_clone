"""
Execution model implementations.

All models implement ExecutionModel.execute(order, current_bar, next_bar) → Optional[Fill].

The execution model pipeline:
  1. Determine raw fill price from bar data.
  2. Apply slippage via the injected SlippageModel.
  3. Create the Fill with the adjusted price.
  4. Compute and assign commission via the injected CommissionModel.

Convention used in the Backtester
----------------------------------
  current_bar  — the bar during which the order was *placed* (on_bar call).
  next_bar     — the bar immediately following (may be None at end of data).

Available models
----------------
NextOpenExecution  — fills at the open of next_bar (standard "next open" fill).
SameBarExecution   — fills at the close of current_bar (immediate same-bar fill).
VWAPExecution      — fills at the VWAP of next_bar: (H + L + C) / 3.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from backtester.interfaces import (
    Bar,
    CommissionModel,
    ExecutionModel,
    Fill,
    Order,
    SlippageModel,
)


def _make_fill(
    order: Order,
    raw_price: float,
    bar: Bar,
    slippage: SlippageModel,
    commission: CommissionModel,
) -> Fill:
    """Shared helper: adjust price, create Fill, then assign commission."""
    price = slippage.adjust(raw_price, order, bar)
    fill = Fill(
        order_id=order.id,
        symbol=order.symbol,
        side=order.side,
        quantity=order.quantity,
        price=price,
        commission=0.0,
        timestamp=bar.timestamp,
    )
    fill.commission = commission.compute(fill)
    return fill


class NextOpenExecution(ExecutionModel):
    """
    Fill at the open price of the *next* bar.

    Returns ``None`` if ``next_bar`` is ``None`` (end of data), leaving the
    order pending until more bars are available or the run ends.

    Parameters
    ----------
    slippage   : SlippageModel applied to the raw open price.
    commission : CommissionModel used to calculate the commission.
    """

    def __init__(self, slippage: SlippageModel, commission: CommissionModel) -> None:
        self._slippage = slippage
        self._commission = commission

    def execute(
        self,
        order: Order,
        current_bar: Bar,
        next_bar: Optional[Bar],
    ) -> Optional[Fill]:
        if next_bar is None:
            return None
        return _make_fill(order, next_bar.open, next_bar, self._slippage, self._commission)


class SameBarExecution(ExecutionModel):
    """
    Fill at the *close* price of the current bar (the bar on which the order
    was placed).

    Always returns a Fill — the fill timestamp matches current_bar.

    Parameters
    ----------
    slippage   : SlippageModel applied to the raw close price.
    commission : CommissionModel used to calculate the commission.
    """

    def __init__(self, slippage: SlippageModel, commission: CommissionModel) -> None:
        self._slippage = slippage
        self._commission = commission

    def execute(
        self,
        order: Order,
        current_bar: Bar,
        next_bar: Optional[Bar],
    ) -> Optional[Fill]:
        return _make_fill(order, current_bar.close, current_bar, self._slippage, self._commission)


class VWAPExecution(ExecutionModel):
    """
    Fill at the VWAP proxy of the *next* bar: (High + Low + Close) / 3.

    Returns ``None`` if ``next_bar`` is ``None`` (end of data).

    Parameters
    ----------
    slippage   : SlippageModel applied to the raw VWAP price.
    commission : CommissionModel used to calculate the commission.
    """

    def __init__(self, slippage: SlippageModel, commission: CommissionModel) -> None:
        self._slippage = slippage
        self._commission = commission

    def execute(
        self,
        order: Order,
        current_bar: Bar,
        next_bar: Optional[Bar],
    ) -> Optional[Fill]:
        if next_bar is None:
            return None
        vwap = (next_bar.high + next_bar.low + next_bar.close) / 3.0
        return _make_fill(order, vwap, next_bar, self._slippage, self._commission)
