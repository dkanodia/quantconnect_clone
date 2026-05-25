"""
OptunaOptimizer — Bayesian / sampler-driven optimizer backed by Optuna.

Concrete implementation of the :class:`~backtester.interfaces.Optimizer` ABC.
Each trial samples a parameter combination via
``ParamSpace.suggest_optuna(trial)``, runs a full backtest, and returns the
objective value to the Optuna study.

After :meth:`optimize` returns, the underlying ``optuna.Study`` is available
as :attr:`OptunaOptimizer.study` and the best ``BacktestResult`` as
:attr:`OptunaOptimizer.best_result` for inspection.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import optuna

from backtester.core.backtester import Backtester
from backtester.exceptions import ConfigurationError, OptimizationError
from backtester.interfaces import BacktestResult, Optimizer, Strategy
from backtester.optimize.grid_search import _validate_backtester_kwargs
from backtester.optimize.param_space import ParamSpace

logger = logging.getLogger(__name__)

_VALID_DIRECTIONS = ("maximize", "minimize")


class OptunaOptimizer(Optimizer):
    """
    Optuna-backed optimizer over a :class:`ParamSpace`.

    Parameters
    ----------
    direction:
        ``"maximize"`` (default) or ``"minimize"`` — the optimisation sense
        of the objective.
    sampler:
        Optional Optuna sampler.  ``None`` uses Optuna's default (TPE).
    pruner:
        Optional Optuna pruner.  ``None`` uses Optuna's default.
    study_name:
        Optional name for the created study.
    show_progress_bar:
        Forwarded to ``study.optimize``.  Defaults to ``False``.

    Attributes
    ----------
    study:
        The ``optuna.Study`` created by the most recent :meth:`optimize` call,
        or ``None`` before the first call.
    best_result:
        The best ``BacktestResult`` from the most recent :meth:`optimize`
        call, or ``None`` before the first call.

    Raises
    ------
    ConfigurationError
        If ``direction`` is not ``"maximize"`` or ``"minimize"``.
    """

    def __init__(
        self,
        direction: str = "maximize",
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None,
        study_name: Optional[str] = None,
        show_progress_bar: bool = False,
    ) -> None:
        if direction not in _VALID_DIRECTIONS:
            raise ConfigurationError(
                f"Invalid direction {direction!r}; expected one of "
                f"{_VALID_DIRECTIONS}."
            )
        self.direction = direction
        self.sampler = sampler
        self.pruner = pruner
        self.study_name = study_name
        self.show_progress_bar = show_progress_bar

        # Whether to clamp Optuna's own logging to WARNING during optimize().
        # Set to False if the caller wants Optuna's INFO logs.
        self.suppress_optuna_logs: bool = True

        self.study: Optional[optuna.Study] = None
        self.best_result: Optional[BacktestResult] = None

    def optimize(
        self,
        strategy_class: type[Strategy],
        param_space: ParamSpace,
        objective: Callable[[BacktestResult], float],
        n_trials: int = 50,
        backtester_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, Any], BacktestResult]:
        """
        Run the Optuna study and return the best parameters and result.

        Parameters
        ----------
        strategy_class:
            Strategy class instantiated as ``strategy_class(**params)`` per trial.
        param_space:
            Search space; ``suggest_optuna(trial)`` samples each trial's combo.
        objective:
            Scores a completed ``BacktestResult``.  Interpreted per ``direction``.
        n_trials:
            Number of Optuna trials to run.  Defaults to ``50``.
        backtester_kwargs:
            Keyword arguments forwarded to ``Backtester`` for every trial.
            Must contain all required components except ``strategy``.

        Returns
        -------
        tuple[dict[str, Any], BacktestResult]
            ``(best_params, best_result)`` for the best completed trial.

        Raises
        ------
        ConfigurationError
            If ``backtester_kwargs`` is missing or lacks required components.
        OptimizationError
            If no trial completed successfully (e.g. all trials pruned).
        """
        _validate_backtester_kwargs(backtester_kwargs)
        assert backtester_kwargs is not None  # narrowed by validation above

        if self.suppress_optuna_logs:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(
            direction=self.direction,
            sampler=self.sampler,
            pruner=self.pruner,
            study_name=self.study_name,
        )
        self.study = study

        # Map completed-trial number → BacktestResult so we can recover the
        # best trial's result after the study finishes.
        results_by_trial: dict[int, BacktestResult] = {}

        def _trial_objective(trial: optuna.Trial) -> float:
            params = param_space.suggest_optuna(trial)
            strategy = strategy_class(**params)
            backtester = Backtester(strategy=strategy, **backtester_kwargs)
            result = backtester.run()
            results_by_trial[trial.number] = result
            return float(objective(result))

        # optuna.exceptions.TrialPruned raised inside the objective is handled
        # natively by Optuna (the trial is marked PRUNED and the study
        # continues).  We do not catch it here.
        study.optimize(
            _trial_objective,
            n_trials=n_trials,
            show_progress_bar=self.show_progress_bar,
        )

        try:
            best_trial = study.best_trial
        except ValueError as exc:
            raise OptimizationError(
                "No Optuna trial completed successfully; cannot determine best "
                "parameters."
            ) from exc

        best_result = results_by_trial.get(best_trial.number)
        if best_result is None:  # pragma: no cover - defensive
            raise OptimizationError(
                "Best trial has no recorded BacktestResult."
            )

        self.best_result = best_result
        return dict(best_trial.params), best_result
