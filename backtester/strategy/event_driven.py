"""
EventDrivenStrategy — base class for bar-by-bar reactive strategies.

Subclass this and override ``on_bar`` (and optionally ``on_start``,
``on_order``, ``on_end``) to build an event-driven strategy.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from backtester.strategy.base import BaseStrategy


class EventDrivenStrategy(BaseStrategy):
    """
    Base class for strategies that react to each bar individually.

    Adds a ``warmup_period`` parameter and a ``bar_index`` counter.  The
    Backtester will not call ``on_bar`` until ``bar_index > warmup_period``
    (i.e. at least ``warmup_period`` bars have been **seen**, not processed).

    All bars, including warmup bars, increment ``bar_index`` so that indicator
    lookback periods are correctly computed.

    Parameters
    ----------
    warmup_period:
        Number of leading bars to skip before ``on_bar`` is first called.
        Defaults to ``0`` (no warmup — ``on_bar`` is called from bar 1).
    **params:
        Arbitrary strategy parameters stored in ``self.params``.

    Example
    -------
    ::

        class SMAStrategy(EventDrivenStrategy):
            def __init__(self, lookback: int = 20):
                super().__init__(warmup_period=lookback, lookback=lookback)
                self.prices: list[float] = []

            def on_bar(self, bar, context=None):
                self.prices.append(bar.close)
                sma = sum(self.prices[-self.params["lookback"]:]) / self.params["lookback"]
                if bar.close > sma:
                    self.buy(bar.symbol, 100)
    """

    def __init__(self, warmup_period: int = 0, **params: Any) -> None:
        super().__init__(**params)
        self.warmup_period: int = warmup_period
        self.bar_index: int = 0

    # ------------------------------------------------------------------
    # Internal API overrides
    # ------------------------------------------------------------------

    def _should_process_bar(self, bar_index: int) -> bool:
        """
        Update ``bar_index`` and return whether ``on_bar`` should be called.

        ``bar_index`` is always updated (even during warmup) so that
        subclasses can rely on it for indicator calculations.

        Returns ``True`` once ``bar_index`` exceeds ``warmup_period``.
        """
        self.bar_index = bar_index
        return bar_index > self.warmup_period

    # ------------------------------------------------------------------
    # Vectorized mode — explicitly unsupported
    # ------------------------------------------------------------------

    def get_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Not supported for event-driven strategies.

        Raises
        ------
        NotImplementedError
            Always.  Use ``VectorizedStrategy`` for signal-based operation.
        """
        raise NotImplementedError(
            f"{type(self).__name__} is an EventDrivenStrategy and does not "
            "support vectorized signal generation via get_signals(). "
            "Override on_bar() to generate orders bar-by-bar, or subclass "
            "VectorizedStrategy instead."
        )
