"""
Parquet-backed DataFeed implementation using Polars lazy scanning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import polars as pl

from backtester.exceptions import DataFeedError, DataNotFoundError
from backtester.interfaces import Bar, DataFeed

from .base import polars_df_to_bars


class ParquetFeed(DataFeed):
    """
    DataFeed that reads OHLCV bars from Parquet files via Polars.

    Accepts either a single ``.parquet`` file or a directory of ``.parquet``
    files.  Within each file, multi-symbol data is supported via an optional
    ``symbol`` column; when that column is absent the symbol is inferred from
    the filename stem.

    All bars are loaded eagerly at construction time and cached in memory.
    Subsequent iterations (including after ``reset()``) replay from the cache
    without re-reading disk.

    Column requirements (case-insensitive)
    ---------------------------------------
    timestamp, open, high, low, close, volume
    [symbol]   — optional; when present, splits the file by ticker

    Parameters
    ----------
    path:
        Path to a single ``.parquet`` file **or** a directory whose
        ``*.parquet`` files are merged and sorted by timestamp.

    Raises
    ------
    DataNotFoundError
        If ``path`` does not exist, or a directory contains no parquet files.
    DataFeedError
        If a file cannot be opened by Polars.
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
        raise DataNotFoundError(
            f"Path '{p}' does not exist or is not a file or directory."
        )

    def _load_directory(self, directory: Path) -> list[Bar]:
        """Load every ``.parquet`` file in *directory*, merge, sort by timestamp."""
        files = sorted(directory.glob("*.parquet"))
        if not files:
            raise DataNotFoundError(
                f"No .parquet files found in directory '{directory}'."
            )
        all_bars: list[Bar] = []
        for f in files:
            all_bars.extend(self._load_file(f))
        return sorted(all_bars, key=lambda b: b.timestamp)

    def _load_file(self, path: Path) -> list[Bar]:
        """
        Read a single parquet file and return a list of Bar objects.

        Uses ``pl.scan_parquet`` (lazy) then ``.collect()`` so Polars can
        push down projections if the file is large.
        """
        try:
            df = pl.scan_parquet(str(path)).collect()
        except Exception as exc:
            raise DataFeedError(
                f"Cannot read parquet file '{path}': {exc}"
            ) from exc

        # Normalise column names to lowercase before any column checks.
        rename_map = {c: c.lower() for c in df.columns if c != c.lower()}
        if rename_map:
            df = df.rename(rename_map)

        if "symbol" in df.columns:
            # Multi-symbol file: split by ticker, convert each slice.
            bars: list[Bar] = []
            for symbol in sorted(df["symbol"].unique().to_list()):
                sym_df = df.filter(pl.col("symbol") == symbol).drop("symbol")
                bars.extend(polars_df_to_bars(sym_df, str(symbol), str(path)))
            return bars

        # Single-symbol file: symbol inferred from filename stem.
        return polars_df_to_bars(df, path.stem, str(path))
