"""
WFOOrchestrator — walk-forward optimization engine.

Splits a feed into ``n_windows`` windows, optimises parameters on each
window's in-sample (IS) portion, evaluates the chosen parameters on the
out-of-sample (OOS) portion, and stitches the OOS equity curves into a single
continuous curve for honest, overfitting-aware performance assessment.

Window modes
------------
``rolling``
    The IS block slides forward with each window (fixed look-back length).
``anchored``
    The IS block always starts at bar 0 and expands; only the OOS block
    slides forward.

In both modes the OOS block for a given window is identical, so the stitched
OOS curve is directly comparable across modes.

Key outputs (see :class:`WFOResult`)
------------------------------------
* per-window summaries (best params, IS/OOS Sharpe, decay),
* the stitched OOS equity curve and its aggregate metrics,
* a single ``robustness_score`` = mean(OOS Sharpe) / mean(IS Sharpe).

Import discipline
-----------------
This module depends only on :class:`~backtester.core.backtester.Backtester`
and :func:`~backtester.analytics.metrics.compute_all` plus the shared
contracts in :mod:`backtester.interfaces` — no direct imports from the
``portfolio``, ``execution``, ``risk``, or ``analytics`` (beyond
``compute_all``) sibling packages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional

import pandas as pd

from backtester.analytics.metrics import compute_all
from backtester.core.backtester import Backtester
from backtester.exceptions import ConfigurationError, WFOError
from backtester.interfaces import (
    BacktestResult,
    Bar,
    DataFeed,
    Optimizer,
    Strategy,
)
from backtester.optimize.param_space import ParamSpace

_DEFAULT_INITIAL_CASH = 100_000.0
_VALID_WINDOW_TYPES = ("rolling", "anchored")


# ---------------------------------------------------------------------------
# Internal DataFeed (not exported)
# ---------------------------------------------------------------------------


class ListFeed(DataFeed):
    """
    Internal ``DataFeed`` that wraps a ``list[Bar]``.

    Used only by :class:`WFOOrchestrator` to build in-sample and out-of-sample
    feeds from pre-loaded bar slices.  Re-iterable: each ``__iter__`` returns a
    fresh iterator, so the same feed can drive many backtests (one per
    optimizer trial) without an explicit reset.
    """

    def __init__(self, bars: list[Bar]) -> None:
        self._bars: list[Bar] = list(bars)

    def __iter__(self) -> Iterator[Bar]:
        """Yield the wrapped bars in order."""
        return iter(self._bars)

    def reset(self) -> None:
        """No-op — the feed is stateless and re-iterable by construction."""

    @property
    def symbols(self) -> list[str]:
        """Return the sorted unique symbols present in the wrapped bars."""
        return sorted({b.symbol for b in self._bars})


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class WFOWindowSummary:
    """
    Summary of a single walk-forward window.

    Attributes
    ----------
    window_index:
        Zero-based position of this window in the walk-forward sequence.
    is_bars:
        Number of in-sample bars used for optimisation.
    oos_bars:
        Number of out-of-sample bars used for evaluation.
    best_params:
        Best parameter dict selected by the optimizer on the IS data.
    is_sharpe:
        Sharpe ratio of the best IS backtest.
    oos_sharpe:
        Sharpe ratio of the OOS backtest run with ``best_params``.
    decay:
        ``is_sharpe - oos_sharpe`` — how much performance degraded OOS.
    oos_result:
        The full OOS ``BacktestResult``.
    """

    window_index: int
    is_bars: int
    oos_bars: int
    best_params: dict[str, Any]
    is_sharpe: float
    oos_sharpe: float
    decay: float
    oos_result: BacktestResult


@dataclass
class WFOResult:
    """
    Aggregate result of a complete walk-forward optimization run.

    Attributes
    ----------
    windows:
        Per-window summaries in walk-forward order.
    oos_equity_curve:
        The OOS equity curves stitched end-to-end into one continuous Series.
    oos_metrics:
        ``compute_all`` metrics computed over the stitched OOS curve and the
        concatenated OOS trades.
    is_sharpes:
        IS Sharpe ratio per window.
    oos_sharpes:
        OOS Sharpe ratio per window.
    decay_scores:
        ``is_sharpe - oos_sharpe`` per window.
    robustness_score:
        ``mean(oos_sharpes) / mean(is_sharpes)``; ``0.0`` if the IS mean is
        zero.
    """

    windows: list[WFOWindowSummary]
    oos_equity_curve: pd.Series
    oos_metrics: dict[str, float]
    is_sharpes: list[float]
    oos_sharpes: list[float]
    decay_scores: list[float]
    robustness_score: float


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class WFOOrchestrator:
    """
    Walk-forward optimization orchestrator.

    Parameters
    ----------
    strategy_class:
        Strategy class to optimise and evaluate.
    param_space:
        Parameter search space passed to the optimizer.
    optimizer:
        Any :class:`~backtester.interfaces.Optimizer` (e.g. ``GridSearch`` or
        ``OptunaOptimizer``) used for the in-sample search.
    objective:
        Objective callable scoring a ``BacktestResult`` (higher is better for
        maximising optimizers).
    n_trials:
        Number of trials per in-sample optimisation.  Defaults to ``50``.
    window_type:
        ``"rolling"`` (default) or ``"anchored"``.
    n_windows:
        Number of walk-forward windows.  Defaults to ``5``.
    is_pct:
        Fraction of each window allocated to in-sample (``0 < is_pct < 1``).
        Defaults to ``0.7``.
    backtester_factory:
        Callable mapping a ``Strategy`` to a fully-configured ``Backtester``.
        Its execution/slippage/commission/risk/reporter/initial_cash
        components are reused for every window; its feed is replaced per
        window with an internal :class:`ListFeed`.  **Required** — ``run``
        raises ``ConfigurationError`` if it is ``None``.

    Raises
    ------
    ConfigurationError
        If ``window_type`` is invalid, ``n_windows < 1``, or ``is_pct`` is not
        strictly between 0 and 1.
    """

    def __init__(
        self,
        strategy_class: type[Strategy],
        param_space: ParamSpace,
        optimizer: Optimizer,
        objective: Callable[[BacktestResult], float],
        n_trials: int = 50,
        window_type: str = "rolling",
        n_windows: int = 5,
        is_pct: float = 0.7,
        backtester_factory: Optional[Callable[[Strategy], Backtester]] = None,
    ) -> None:
        if window_type not in _VALID_WINDOW_TYPES:
            raise ConfigurationError(
                f"Invalid window_type {window_type!r}; expected one of "
                f"{_VALID_WINDOW_TYPES}."
            )
        if n_windows < 1:
            raise ConfigurationError(
                f"n_windows must be >= 1, got {n_windows}."
            )
        if not (0.0 < is_pct < 1.0):
            raise ConfigurationError(
                f"is_pct must be strictly between 0 and 1, got {is_pct}."
            )

        self._strategy_class = strategy_class
        self._param_space = param_space
        self._optimizer = optimizer
        self._objective = objective
        self._n_trials = n_trials
        self._window_type = window_type
        self._n_windows = n_windows
        self._is_pct = is_pct
        self._backtester_factory = backtester_factory

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, feed: DataFeed) -> WFOResult:
        """
        Execute the walk-forward optimization and return a :class:`WFOResult`.

        Parameters
        ----------
        feed:
            Source feed providing the full bar history to partition.

        Returns
        -------
        WFOResult
            Per-window summaries, stitched OOS equity curve, aggregate OOS
            metrics, IS/OOS Sharpe series, decay scores, and robustness score.

        Raises
        ------
        ConfigurationError
            If no ``backtester_factory`` was supplied.
        WFOError
            If the feed is empty or cannot be partitioned into the requested
            number of windows.
        """
        if self._backtester_factory is None:
            raise ConfigurationError(
                "WFOOrchestrator.run requires a backtester_factory."
            )

        bars: list[Bar] = list(feed)
        if not bars:
            raise WFOError("Feed produced no bars; cannot run walk-forward.")

        windows = self._split_windows(bars)
        components, initial_cash = self._template_components()

        summaries: list[WFOWindowSummary] = []
        is_sharpes: list[float] = []
        oos_sharpes: list[float] = []
        decay_scores: list[float] = []
        oos_curves: list[pd.Series] = []
        oos_trades: list[pd.DataFrame] = []

        for idx, (is_bars, oos_bars) in enumerate(windows):
            # --- In-sample optimisation ---
            is_kwargs = {**components, "feed": ListFeed(is_bars)}
            best_params, is_result = self._optimizer.optimize(
                self._strategy_class,
                self._param_space,
                self._objective,
                self._n_trials,
                is_kwargs,
            )
            is_sharpe = float(is_result.metrics.get("sharpe_ratio", 0.0))

            # --- Out-of-sample evaluation ---
            oos_kwargs = {**components, "feed": ListFeed(oos_bars)}
            oos_strategy = self._strategy_class(**best_params)
            oos_result = Backtester(strategy=oos_strategy, **oos_kwargs).run()
            oos_sharpe = float(oos_result.metrics.get("sharpe_ratio", 0.0))

            decay = is_sharpe - oos_sharpe

            summaries.append(
                WFOWindowSummary(
                    window_index=idx,
                    is_bars=len(is_bars),
                    oos_bars=len(oos_bars),
                    best_params=best_params,
                    is_sharpe=is_sharpe,
                    oos_sharpe=oos_sharpe,
                    decay=decay,
                    oos_result=oos_result,
                )
            )
            is_sharpes.append(is_sharpe)
            oos_sharpes.append(oos_sharpe)
            decay_scores.append(decay)
            oos_curves.append(oos_result.equity_curve)
            oos_trades.append(oos_result.trades)

        stitched = self._stitch_curves(oos_curves, initial_cash)
        oos_metrics = self._aggregate_metrics(stitched, oos_trades)
        robustness = self._robustness(is_sharpes, oos_sharpes)

        return WFOResult(
            windows=summaries,
            oos_equity_curve=stitched,
            oos_metrics=oos_metrics,
            is_sharpes=is_sharpes,
            oos_sharpes=oos_sharpes,
            decay_scores=decay_scores,
            robustness_score=robustness,
        )

    # ------------------------------------------------------------------
    # Window splitting
    # ------------------------------------------------------------------

    def _split_windows(
        self, bars: list[Bar]
    ) -> list[tuple[list[Bar], list[Bar]]]:
        """
        Partition *bars* into ``n_windows`` (IS, OOS) segment pairs.

        The OOS segment of each window is identical between rolling and
        anchored modes; only the IS segment differs:

        * rolling  → IS = ``bars[w_start:split]``
        * anchored → IS = ``bars[0:split]``

        Raises
        ------
        WFOError
            If there are too few bars to form the requested windows.
        """
        n = len(bars)
        w = self._n_windows
        window_size = n // w
        if window_size < 2:
            raise WFOError(
                f"Not enough bars ({n}) for {w} windows; need at least "
                f"{2 * w}."
            )

        windows: list[tuple[list[Bar], list[Bar]]] = []
        for i in range(w):
            w_start = i * window_size
            w_end = n if i == w - 1 else (i + 1) * window_size

            is_count = int(window_size * self._is_pct)
            is_count = max(1, min(window_size - 1, is_count))
            split = w_start + is_count

            if self._window_type == "rolling":
                is_bars = bars[w_start:split]
            else:  # anchored
                is_bars = bars[0:split]
            oos_bars = bars[split:w_end]

            if not is_bars or not oos_bars:
                raise WFOError(
                    f"Window {i} produced an empty IS or OOS segment "
                    f"(is={len(is_bars)}, oos={len(oos_bars)})."
                )
            windows.append((is_bars, oos_bars))

        return windows

    # ------------------------------------------------------------------
    # Template component extraction
    # ------------------------------------------------------------------

    def _template_components(self) -> tuple[dict[str, Any], float]:
        """
        Build a template Backtester via the factory and extract its components.

        A probe strategy is created from the first grid combination so the
        factory can run even when the strategy requires constructor params.
        The feed is intentionally excluded — the orchestrator supplies a
        per-window :class:`ListFeed` instead.

        Returns
        -------
        tuple[dict[str, Any], float]
            ``(components, initial_cash)`` where *components* is the kwargs
            dict (minus ``feed``) shared by every backtest.

        Raises
        ------
        ConfigurationError
            If the factory does not return a ``Backtester`` instance.
        """
        assert self._backtester_factory is not None  # checked in run()

        probe_params = self._param_space.grid_combinations()[0]
        probe_strategy = self._strategy_class(**probe_params)
        template = self._backtester_factory(probe_strategy)

        try:
            components: dict[str, Any] = {
                "execution": template._execution,
                "slippage": template._slippage,
                "commission": template._commission,
                "risk": template._risk,
                "reporter": template._reporter,
                "initial_cash": template._initial_cash,
            }
            initial_cash = float(template._initial_cash)
        except AttributeError as exc:
            raise ConfigurationError(
                "backtester_factory must return a Backtester instance."
            ) from exc

        return components, initial_cash

    # ------------------------------------------------------------------
    # Stitching & aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _stitch_curves(
        curves: list[pd.Series], initial_cash: float
    ) -> pd.Series:
        """
        Chain per-window OOS equity curves into one continuous Series.

        The first window starts at *initial_cash*; each subsequent window is
        scaled so its first point continues from the previous window's last
        equity value.  The result is indexed by the original OOS timestamps
        (monotonic, no overlaps).
        """
        segments: list[pd.Series] = []
        running = float(initial_cash)

        for curve in curves:
            if curve is None or len(curve) == 0:
                continue
            base = float(curve.iloc[0])
            rel = curve / base if base != 0.0 else (curve * 0.0 + 1.0)
            segment = rel * running
            segments.append(segment)
            running = float(segment.iloc[-1])

        if not segments:
            return pd.Series(dtype=float, name="equity")

        stitched = pd.concat(segments)
        stitched.name = "equity"
        return stitched

    @staticmethod
    def _aggregate_metrics(
        stitched: pd.Series, oos_trades: list[pd.DataFrame]
    ) -> dict[str, float]:
        """Run ``compute_all`` over the stitched curve and concatenated trades."""
        if oos_trades:
            trades = pd.concat(oos_trades, ignore_index=True)
        else:
            trades = pd.DataFrame()

        synthetic = BacktestResult(
            equity_curve=stitched,
            trades=trades,
            metrics={},
            params={},
            strategy_name="WFO_stitched_OOS",
        )
        return compute_all(synthetic)

    @staticmethod
    def _robustness(
        is_sharpes: list[float], oos_sharpes: list[float]
    ) -> float:
        """Return mean(OOS Sharpe) / mean(IS Sharpe); ``0.0`` if IS mean is 0."""
        if not is_sharpes or not oos_sharpes:
            return 0.0
        mean_is = sum(is_sharpes) / len(is_sharpes)
        mean_oos = sum(oos_sharpes) / len(oos_sharpes)
        if mean_is == 0.0:
            return 0.0
        return float(mean_oos / mean_is)
