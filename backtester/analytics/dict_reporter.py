"""
DictReporter — serialises a BacktestResult to a JSON-safe Python dictionary.

All values in the returned dict are plain Python scalars (``float``, ``int``,
``str``, ``bool``, ``None``, ``list``, ``dict``).  No ``numpy`` dtypes, no
``pandas`` objects — equity_curve timestamps are ISO 8601 strings.

Returned structure::

    {
        "run_id":        str,
        "strategy_name": str,
        "owner":         str,
        "created_at":    str (ISO 8601),
        "params":        dict,
        "metrics":       dict[str, float | int],  # from compute_all()
        "equity_curve":  dict[str, float],        # {timestamp_iso: value}
        "num_bars":      int,
    }

Usage
-----
>>> reporter = DictReporter()
>>> d = reporter.report(result)
>>> import json; json.dumps(d)   # always succeeds
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from backtester.analytics.metrics import compute_all
from backtester.interfaces import BacktestResult, Reporter


def _to_python(value: Any) -> Any:
    """
    Coerce *value* to a plain Python type recursively.

    Handles numpy scalars (expose ``.item()``), ``pandas.Timestamp``,
    ``datetime.datetime``, ``dict``, and ``list``.  Everything else is
    returned unchanged (assumed already JSON-safe).
    """
    # numpy scalars expose .item() which returns the native Python equivalent
    if hasattr(value, "item") and callable(value.item):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    # datetime.datetime / date (not pd.Timestamp — that's handled above)
    if hasattr(value, "isoformat") and callable(value.isoformat):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _to_python(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_python(v) for v in value]
    return value


class DictReporter(Reporter):
    """
    Serialise a ``BacktestResult`` to a JSON-safe Python ``dict``.

    Implements the :class:`~backtester.interfaces.Reporter` interface.

    The ``metrics`` key contains the output of
    :func:`~backtester.analytics.metrics.compute_all`, which always uses
    plain Python ``float`` / ``int`` values.

    The ``equity_curve`` key maps ISO 8601 timestamp strings to ``float``
    equity values so the dict can be passed directly to ``json.dumps``
    without a custom encoder.
    """

    def report(self, result: BacktestResult) -> dict[str, Any]:
        """
        Convert *result* to a JSON-serialisable dictionary.

        Parameters
        ----------
        result : BacktestResult
            The completed backtest result.

        Returns
        -------
        dict[str, Any]
            Plain Python dict — always safe to pass to ``json.dumps``.
        """
        # -- equity_curve: datetime index → ISO 8601 string keys --------
        equity_curve: dict[str, float] = {}
        for ts, val in result.equity_curve.items():
            key = (
                ts.isoformat()
                if isinstance(ts, pd.Timestamp)
                else str(ts)
            )
            equity_curve[key] = float(val)

        # -- params: coerce any numpy types ------------------------------
        safe_params: dict[str, Any] = _to_python(result.params)  # type: ignore[assignment]

        # -- created_at --------------------------------------------------
        created_at: str = (
            result.created_at.isoformat()
            if hasattr(result.created_at, "isoformat")
            else str(result.created_at)
        )

        return {
            "run_id": result.run_id,
            "strategy_name": result.strategy_name,
            "owner": result.owner,
            "created_at": created_at,
            "params": safe_params,
            "metrics": compute_all(result),
            "equity_curve": equity_curve,
            "num_bars": len(result.equity_curve),
        }
