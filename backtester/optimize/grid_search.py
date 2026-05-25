"""
GridSearch — exhaustive (or randomly-sampled) grid-search optimizer.

Concrete implementation of the :class:`~backtester.interfaces.Optimizer` ABC.
Enumerates every combination produced by ``ParamSpace.grid_combinations()``,
runs a full backtest for each, scores it with the supplied objective, and
returns the best ``(params, result)`` pair.

When ``n_trials`` is supplied and is smaller than the total number of
combinations, a random subset of that size is sampled without replacement
instead of running the full grid.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Callable, Optional

from backtester.core.backtester import Backtester
from backtester.exceptions import ConfigurationError, OptimizationError
from backtester.interfaces import BacktestResult, Optimizer, Strategy
from backtester.optimize.param_space import ParamSpace

logger = logging.getLogger(__name__)

# Backtester components that must be present in ``backtester_kwargs``
# (``strategy`` is supplied per-trial by the optimizer, so it is excluded).
_REQUIRED_BT_KEYS: tuple[str, ...] = (
    "feed",
    "execution",
    "slippage",
    "commission",
    "risk",
    "reporter",
)


class GridSearch(Optimizer):
    """
    Grid-search optimizer over a :class:`ParamSpace`.

    Implements :meth:`optimize`.  Higher objective values are better; the
    optimizer keeps the single best-scoring trial.
    """

    def optimize(
        self,
        strategy_class: type[Strategy],
        param_space: ParamSpace,
        objective: Callable[[BacktestResult], float],
        n_trials: Optional[int] = None,
        backtester_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, Any], BacktestResult]:
        """
        Run the grid search and return the best parameters and result.

        Parameters
        ----------
        strategy_class:
            Strategy class instantiated as ``strategy_class(**params)`` per trial.
        param_space:
            Search space; ``grid_combinations()`` supplies the candidate combos.
        objective:
            Scores a completed ``BacktestResult``; higher is better.
        n_trials:
            If set and smaller than the total combinations, a random subset of
            this size is evaluated (sampled without replacement).  If ``None``
            the full grid is evaluated.
        backtester_kwargs:
            Keyword arguments forwarded to ``Backtester`` for every trial.
            Must contain all required components except ``strategy``:
            ``feed``, ``execution``, ``slippage``, ``commission``, ``risk``,
            ``reporter`` (``initial_cash`` / ``debug`` optional).

        Returns
        -------
        tuple[dict[str, Any], BacktestResult]
            ``(best_params, best_result)`` for the highest objective value.

        Raises
        ------
        ConfigurationError
            If ``backtester_kwargs`` is missing or lacks required components.
        OptimizationError
            If no trials could be evaluated (empty combination set).
        """
        _validate_backtester_kwargs(backtester_kwargs)
        assert backtester_kwargs is not None  # narrowed by validation above

        combos = param_space.grid_combinations()
        if not combos:
            raise OptimizationError("ParamSpace produced no grid combinations.")

        if n_trials is not None and 0 <= n_trials < len(combos):
            combos = random.sample(combos, n_trials)

        total = len(combos)
        best_score = float("-inf")
        best_params: Optional[dict[str, Any]] = None
        best_result: Optional[BacktestResult] = None

        for i, params in enumerate(combos, start=1):
            strategy = strategy_class(**params)
            backtester = Backtester(strategy=strategy, **backtester_kwargs)
            result = backtester.run()
            score = float(objective(result))

            logger.info(
                "GridSearch trial %d/%d: params=%s objective=%.4f",
                i, total, params, score,
            )

            if score > best_score:
                best_score = score
                best_params = params
                best_result = result

        if best_params is None or best_result is None:
            raise OptimizationError("GridSearch evaluated zero trials.")

        return best_params, best_result


def _validate_backtester_kwargs(
    backtester_kwargs: Optional[dict[str, Any]],
) -> None:
    """
    Raise ``ConfigurationError`` if required Backtester components are missing.

    Parameters
    ----------
    backtester_kwargs:
        The kwargs dict (or ``None``) supplied to :meth:`GridSearch.optimize`.

    Raises
    ------
    ConfigurationError
        If ``backtester_kwargs`` is ``None``/empty or lacks any required key.
    """
    if not backtester_kwargs:
        raise ConfigurationError(
            "backtester_kwargs is required and must contain: "
            f"{', '.join(_REQUIRED_BT_KEYS)}."
        )
    missing = [k for k in _REQUIRED_BT_KEYS if k not in backtester_kwargs]
    if missing:
        raise ConfigurationError(
            f"backtester_kwargs is missing required components: {missing}."
        )
