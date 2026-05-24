"""
VectorizedStrategy — base class for signal-based fast-mode strategies.

Subclass this and override ``get_signals`` to build a strategy optimised for
parameter sweeps and walk-forward optimisation.  The Backtester collects the
full bar history, calls ``get_signals`` once, and converts the returned signal
Series into orders in bulk.

Event-driven hooks (``on_bar``, ``on_start``, etc.) are *not* called in
vectorized mode and raise ``NotImplementedError`` to surface accidental usage.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from backtester.interfaces import Bar
from backtester.strategy.base import BaseStrategy


class VectorizedStrategy(BaseStrategy):
    """
    Base class for strategies that generate signals over the full price history.

    Subclasses must override ``get_signals``; all other hooks are unsupported
    and raise ``NotImplementedError`` to prevent accidental event-driven usage.

    Signal convention
    -----------------
    ``get_signals`` should return a ``pd.Series`` indexed by timestamp with
    float values in ``{-1, 0, 1}``:

    *  ``1``  → go long
    *  ``0``  → flat / no position
    * ``-1``  → go short

    Fractional values are allowed and interpreted as position-sizing weights.

    Parameters
    ----------
    **params:
        Arbitrary strategy parameters stored in ``self.params``.

    Example
    -------
    ::

        class MACrossStrategy(VectorizedStrategy):
            def __init__(self, fast: int = 10, slow: int = 50):
                super().__init__(fast=fast, slow=slow)

            def get_signals(self, data: pd.DataFrame) -> pd.Series:
                fast_ma = data["close"].rolling(self.params["fast"]).mean()
                slow_ma = data["close"].rolling(self.params["slow"]).mean()
                return (fast_ma > slow_ma).astype(float) * 2 - 1
    """

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)

    # ------------------------------------------------------------------
    # Internal API overrides
    # ------------------------------------------------------------------

    def _is_vectorized(self) -> bool:
        """Return ``True`` — signals this strategy uses the vectorized loop."""
        return True

    # ------------------------------------------------------------------
    # Event-driven hooks — explicitly unsupported
    # ------------------------------------------------------------------

    def on_bar(self, bar: Bar, context: Any = None) -> None:
        """
        Not supported for vectorized strategies.

        Raises
        ------
        NotImplementedError
            Always.  Override ``get_signals()`` instead.
        """
        raise NotImplementedError(
            f"{type(self).__name__} is a VectorizedStrategy and does not "
            "support event-driven bar-by-bar processing via on_bar(). "
            "Override get_signals() to return a signal Series, or subclass "
            "EventDrivenStrategy instead."
        )

    # ------------------------------------------------------------------
    # Vectorized interface — must be overridden
    # ------------------------------------------------------------------

    def get_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate a signal Series over the full bar history.

        Parameters
        ----------
        data:
            DataFrame of OHLCV bars indexed by timestamp with columns
            ``open``, ``high``, ``low``, ``close``, ``volume``.

        Returns
        -------
        pd.Series
            Float signals indexed by timestamp.  Values should be in
            ``{-1, 0, 1}`` where ``1`` = long, ``0`` = flat, ``-1`` = short.

        Raises
        ------
        NotImplementedError
            If this method is not overridden by the subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override get_signals() to return "
            "a signal Series.  See VectorizedStrategy docstring for the "
            "expected return format."
        )
