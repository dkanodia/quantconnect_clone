"""
Optimization layer — parameter search and walk-forward analysis.

Public API
----------
Parameter space:
    ParamSpace, IntParam, FloatParam, CategoricalParam, ParamDef

Optimizers (implement the Optimizer ABC):
    GridSearch       — exhaustive / random grid search
    OptunaOptimizer  — Optuna sampler-driven search

Walk-forward:
    WFOOrchestrator  — walk-forward optimization engine
    WFOResult        — aggregate run result
    WFOWindowSummary — per-window summary

``ListFeed`` (used internally by ``WFOOrchestrator``) is intentionally **not**
exported.
"""

from backtester.optimize.grid_search import GridSearch
from backtester.optimize.optuna_optimizer import OptunaOptimizer
from backtester.optimize.param_space import (
    CategoricalParam,
    FloatParam,
    IntParam,
    ParamDef,
    ParamSpace,
)
from backtester.optimize.wfo import WFOOrchestrator, WFOResult, WFOWindowSummary

__all__ = [
    # param space
    "ParamSpace",
    "IntParam",
    "FloatParam",
    "CategoricalParam",
    "ParamDef",
    # optimizers
    "GridSearch",
    "OptunaOptimizer",
    # walk-forward
    "WFOOrchestrator",
    "WFOResult",
    "WFOWindowSummary",
]
