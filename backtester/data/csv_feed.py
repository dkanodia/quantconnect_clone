"""
CSV-backed DataFeed implementation using Polars.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import polars as pl

from backtester.exceptions import DataFeedError, DataNotFoundError
from backtester.interfaces import Bar, DataFeed

from .base import polars_df_to_bars


class CSVFeed(DataFeed):
    """
    DataFeed that reads OHLCV bars from CSV files via Polars.

    Accepts a single CSV file or a directory of CSV files.  Polars'
    ``try_parse_dates=True`` option handles the most common date and datetime
    string formats automatically; integer Unix epoch columns (seconds) are
    also supported via the shared ``normalize_timestamp_series`` utility.

    An optional ``symbol`` column enables multi-ticker files; when absent the
    symbol is inferred from the filename stem.

    Column requirements (case-insensitive)
    ---------------------------------------
    timestamp, open, high, low, close, volume
    [symbol]   — optional

    Timestamp formats handled automatically
    ----------------------------------------
    - ISO 8601 strings  (``2024-01-02T09:30:00``, ``2024-01-02T09:30:00Z``)
    - Date strings      (``2024-01-02``, ``01/02/2024``)
    - Unix epoch integers (seconds)

    Parameters
    ----------
    path:
        Path to a single CSV file **or** a directory whose ``*.csv`` files
        are merged and sorted by timestamp.

    Raises
    ------
    DataNotFoundError
        If ``path`` does not exist, or a directory contains no CSV files.
    DataFeedError
        If a file cannot be read by Polars.
    DataCorruptionError
        If required columns are absent or timestamps are unparseable.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._bars: list[Bar] = self._load()

    # ------------------------------------------------------------------
    # DataFeed interface
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[Bar]:
        """Yield all bars in ascending timestamp order."""
        return iter(self._bars)

    def reset(self) -> None:
        """
        Reset the feed to its initial state.

        Because bars are fully cached, each call to ``__iter__`` already
        starts from the first bar.  This method is a deliberate no-op.
        """

    @property
    def symbols(self) -> list[str]:
        """Return a sorted list of unique ticker symbols present in this feed."""
        return sorted({b.symbol for b in self._bars})

    # ------------------------------------------------------------------
    # Private loading helpers
    # ------------------------------------------------------------------

    def _load(self) -> list[Bar]:
        """Dispatch to file or directory loading; return sorted bars."""
        p = self._path
        if p.is_file():
            return self._load_file(p)
        if p.is_dir():
            return self._load_directory(p)
        raise DataNotFoundError(f"Path '{p}' does not exist.")

    def _load_directory(self, directory: Path) -> list[Bar]:
        """Load every ``.csv`` file in *directory*, merge, sort by timestamp."""
        files = sorted(directory.glob("*.csv"))
        if not files:
            raise DataNotFoundError(
                f"No .csv files found in directory '{directory}'."
            )
        all_bars: list[Bar] = []
        for f in files:
            all_bars.extend(self._load_file(f))
        return sorted(all_bars, key=lambda b: b.timestamp)

    def _load_file(self, path: Path) -> list[Bar]:
        """
        Read a single CSV file and return a list of Bar objects.

        ``try_parse_dates=True`` instructs Polars to attempt automatic
        date/datetime parsing for any column whose values look like dates.
        Integer timestamp columns are handled downstream by
        ``normalize_timestamp_series``.
        """
        try:
            df = pl.read_csv(
                str(path),
                try_parse_dates=True,
                infer_schema_length=1000,
            )
        except Exception as exc:
            raise DataFeedError(
                f"Cannot read CSV file '{path}': {exc}"
            ) from exc

        # Normalise column names to lowercase.
        rename_map = {c: c.lower() for c in df.columns if c != c.lower()}
        if rename_map:
            df = df.rename(rename_map)

        if "symbol" in df.columns:
            bars: list[Bar] = []
            for symbol in sorted(df["symbol"].unique().to_list()):
                sym_df = df.filter(pl.col("symbol") == symbol).drop("symbol")
                bars.extend(polars_df_to_bars(sym_df, str(symbol), str(path)))
            return bars

        return polars_df_to_bars(df, path.stem, str(path))
