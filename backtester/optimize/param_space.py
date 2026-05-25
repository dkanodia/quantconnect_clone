"""
ParamSpace — typed parameter-search-space definitions.

Shared by :class:`~backtester.optimize.grid_search.GridSearch` (exhaustive /
random sampling) and
:class:`~backtester.optimize.optuna_optimizer.OptunaOptimizer` (Bayesian
sampling).

Three parameter kinds are supported:

``IntParam``
    Integer range ``[low, high]`` inclusive, sampled in increments of ``step``.
``FloatParam``
    Continuous range ``[low, high]``; ``log=True`` requests log-scale
    sampling under Optuna and requires a strictly positive ``low``.
``CategoricalParam``
    An explicit list of choices of any type.

A ``ParamSpace`` bundles a list of these definitions and knows how to:

* enumerate every grid combination (:meth:`ParamSpace.grid_combinations`),
* sample one combination from an Optuna trial
  (:meth:`ParamSpace.suggest_optuna`), and
* validate a concrete parameter dict against the declared bounds
  (:meth:`ParamSpace.validate`).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Union

import numpy as np

from backtester.exceptions import ConfigurationError

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    import optuna

# Number of points sampled across a FloatParam range for the fixed grid.
_FLOAT_GRID_POINTS = 10


@dataclass
class IntParam:
    """
    Integer search dimension over the inclusive range ``[low, high]``.

    Parameters
    ----------
    name:
        Strategy constructor keyword this parameter maps to.
    low:
        Inclusive lower bound.
    high:
        Inclusive upper bound.
    step:
        Positive grid/sampling increment.  Defaults to ``1``.
    """

    name: str
    low: int
    high: int
    step: int = 1


@dataclass
class FloatParam:
    """
    Continuous search dimension over the range ``[low, high]``.

    Parameters
    ----------
    name:
        Strategy constructor keyword this parameter maps to.
    low:
        Lower bound.
    high:
        Upper bound.
    log:
        When ``True``, Optuna samples on a logarithmic scale.  Requires a
        strictly positive ``low``.  Defaults to ``False``.
    """

    name: str
    low: float
    high: float
    log: bool = False


@dataclass
class CategoricalParam:
    """
    Discrete search dimension over an explicit list of choices.

    Parameters
    ----------
    name:
        Strategy constructor keyword this parameter maps to.
    choices:
        Non-empty list of candidate values (any type).
    """

    name: str
    choices: list[Any]


# A single parameter definition — one of the three concrete kinds above.
ParamDef = Union[IntParam, FloatParam, CategoricalParam]


@dataclass
class ParamSpace:
    """
    A typed container describing a parameter search space.

    Parameters
    ----------
    params:
        Non-empty list of :data:`ParamDef` definitions.

    Notes
    -----
    Bounds are validated lazily — the first call to
    :meth:`grid_combinations` or :meth:`suggest_optuna` raises
    ``ConfigurationError`` if the space is empty or any definition has
    invalid bounds.
    """

    params: list[ParamDef]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def grid_combinations(self) -> list[dict[str, Any]]:
        """
        Return every parameter combination for an exhaustive grid search.

        Per-dimension value sets:

        * ``IntParam``          → ``range(low, high + 1, step)``
        * ``FloatParam``        → ``np.linspace(low, high, num=10)``
        * ``CategoricalParam``  → ``choices``

        Returns
        -------
        list[dict[str, Any]]
            The Cartesian product of all dimensions, one dict per combination.

        Raises
        ------
        ConfigurationError
            If the space is empty or any definition has invalid bounds.
        """
        self._validate_space()

        names: list[str] = []
        value_lists: list[list[Any]] = []
        for p in self.params:
            names.append(p.name)
            value_lists.append(self._grid_values(p))

        return [
            dict(zip(names, combo))
            for combo in itertools.product(*value_lists)
        ]

    def suggest_optuna(self, trial: "optuna.Trial") -> dict[str, Any]:
        """
        Sample one parameter dict from an Optuna trial.

        Maps each definition to the matching ``trial.suggest_*`` call:

        * ``IntParam``          → ``trial.suggest_int(name, low, high, step=step)``
        * ``FloatParam``        → ``trial.suggest_float(name, low, high, log=log)``
        * ``CategoricalParam``  → ``trial.suggest_categorical(name, choices)``

        Parameters
        ----------
        trial:
            The active ``optuna.Trial`` to draw suggestions from.

        Returns
        -------
        dict[str, Any]
            One sampled parameter combination.

        Raises
        ------
        ConfigurationError
            If the space is empty or any definition has invalid bounds.
        """
        self._validate_space()

        out: dict[str, Any] = {}
        for p in self.params:
            if isinstance(p, IntParam):
                out[p.name] = trial.suggest_int(
                    p.name, p.low, p.high, step=p.step
                )
            elif isinstance(p, FloatParam):
                out[p.name] = trial.suggest_float(
                    p.name, p.low, p.high, log=p.log
                )
            elif isinstance(p, CategoricalParam):
                out[p.name] = trial.suggest_categorical(p.name, p.choices)
            else:  # pragma: no cover - guarded by _validate_space
                raise ConfigurationError(
                    f"Unknown ParamDef type: {type(p).__name__}"
                )
        return out

    def validate(self, params: dict[str, Any]) -> None:
        """
        Raise ``ConfigurationError`` if *params* does not satisfy the space.

        Checks that every declared parameter is present and that each value
        lies within the declared bounds (range membership for numeric kinds,
        choice membership for categorical kinds).

        Parameters
        ----------
        params:
            A concrete parameter dict (e.g. one returned by
            :meth:`grid_combinations`).

        Raises
        ------
        ConfigurationError
            If a parameter is missing or out of bounds.
        """
        self._validate_space()

        for p in self.params:
            if p.name not in params:
                raise ConfigurationError(
                    f"Parameter '{p.name}' missing from params dict."
                )
            value = params[p.name]

            if isinstance(p, IntParam):
                if not (p.low <= value <= p.high):
                    raise ConfigurationError(
                        f"Parameter '{p.name}'={value} outside "
                        f"[{p.low}, {p.high}]."
                    )
            elif isinstance(p, FloatParam):
                if not (p.low <= value <= p.high):
                    raise ConfigurationError(
                        f"Parameter '{p.name}'={value} outside "
                        f"[{p.low}, {p.high}]."
                    )
            elif isinstance(p, CategoricalParam):
                if value not in p.choices:
                    raise ConfigurationError(
                        f"Parameter '{p.name}'={value!r} not in "
                        f"choices {p.choices!r}."
                    )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_space(self) -> None:
        """Validate the space definition itself (non-empty, valid bounds)."""
        if not self.params:
            raise ConfigurationError(
                "ParamSpace must contain at least one parameter definition."
            )

        for p in self.params:
            if not getattr(p, "name", None):
                raise ConfigurationError("Every ParamDef must have a non-empty name.")

            if isinstance(p, IntParam):
                if p.low > p.high:
                    raise ConfigurationError(
                        f"IntParam '{p.name}': low ({p.low}) > high ({p.high})."
                    )
                if p.step <= 0:
                    raise ConfigurationError(
                        f"IntParam '{p.name}': step must be positive, got {p.step}."
                    )
            elif isinstance(p, FloatParam):
                if p.low > p.high:
                    raise ConfigurationError(
                        f"FloatParam '{p.name}': low ({p.low}) > high ({p.high})."
                    )
                if p.log and p.low <= 0:
                    raise ConfigurationError(
                        f"FloatParam '{p.name}': log-scale sampling requires "
                        f"low > 0, got {p.low}."
                    )
            elif isinstance(p, CategoricalParam):
                if not p.choices:
                    raise ConfigurationError(
                        f"CategoricalParam '{p.name}': choices must be non-empty."
                    )
            else:
                raise ConfigurationError(
                    f"Unknown ParamDef type: {type(p).__name__}"
                )

    @staticmethod
    def _grid_values(p: ParamDef) -> list[Any]:
        """Return the discrete grid value list for a single dimension."""
        if isinstance(p, IntParam):
            return list(range(p.low, p.high + 1, p.step))
        if isinstance(p, FloatParam):
            return [float(x) for x in np.linspace(p.low, p.high, num=_FLOAT_GRID_POINTS)]
        if isinstance(p, CategoricalParam):
            return list(p.choices)
        raise ConfigurationError(  # pragma: no cover - guarded by _validate_space
            f"Unknown ParamDef type: {type(p).__name__}"
        )
