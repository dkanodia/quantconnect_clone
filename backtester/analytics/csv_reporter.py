"""
CSVReporter — writes a BacktestResult to two CSV files on disk.

Files created
-------------
``{output_dir}/{run_id}_metrics.csv``
    Two-column CSV with headers ``metric,value`` — one row per key returned
    by :func:`~backtester.analytics.metrics.compute_all`.

``{output_dir}/{run_id}_trades.csv``
    Full ``result.trades`` DataFrame written as-is (one row per completed
    trade).

The reporter creates *output_dir* (and any missing parents) if it does not
already exist.  Any ``OSError`` encountered during directory creation or file
writing is caught and re-raised as a
:class:`~backtester.exceptions.BacktestError`.

Usage
-----
>>> reporter = CSVReporter(output_dir="results/")
>>> out_dir = reporter.report(result)   # returns Path to the output directory
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

from backtester.analytics.metrics import compute_all
from backtester.exceptions import BacktestError
from backtester.interfaces import BacktestResult, Reporter


class CSVReporter(Reporter):
    """
    Write a ``BacktestResult`` to CSV files in a specified directory.

    Parameters
    ----------
    output_dir : str or Path
        Directory where the two CSV files will be written.  Created
        (including intermediate parents) if it does not already exist.
        Defaults to ``"./backtest_results"``.

    Raises
    ------
    BacktestError
        If the directory cannot be created or the files cannot be written
        (wraps the underlying ``OSError``).
    """

    def __init__(
        self, output_dir: Union[str, Path] = "./backtest_results"
    ) -> None:
        self._output_dir = Path(output_dir)

    def report(self, result: BacktestResult) -> Path:
        """
        Write metrics and trades CSVs for *result*.

        Files written:

        * ``<output_dir>/<run_id>_metrics.csv`` — columns: ``metric``, ``value``
        * ``<output_dir>/<run_id>_trades.csv``  — one row per completed trade

        Parameters
        ----------
        result : BacktestResult
            The completed backtest result.

        Returns
        -------
        pathlib.Path
            Path to *output_dir* (the directory containing both files).

        Raises
        ------
        BacktestError
            If the directory cannot be created or the files cannot be written.
        """
        # Ensure output directory exists
        try:
            self._output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise BacktestError(
                f"CSVReporter: cannot create output directory "
                f"'{self._output_dir}': {exc}",
                strategy_name=result.strategy_name,
            ) from exc

        metrics_path = self._output_dir / f"{result.run_id}_metrics.csv"
        trades_path = self._output_dir / f"{result.run_id}_trades.csv"

        # Write metrics CSV
        try:
            metrics = compute_all(result)
            metrics_df = pd.DataFrame(
                list(metrics.items()), columns=["metric", "value"]
            )
            metrics_df.to_csv(metrics_path, index=False)
        except OSError as exc:
            raise BacktestError(
                f"CSVReporter: cannot write metrics CSV to '{metrics_path}': {exc}",
                strategy_name=result.strategy_name,
            ) from exc

        # Write trades CSV
        try:
            result.trades.to_csv(trades_path, index=False)
        except OSError as exc:
            raise BacktestError(
                f"CSVReporter: cannot write trades CSV to '{trades_path}': {exc}",
                strategy_name=result.strategy_name,
            ) from exc

        return self._output_dir
