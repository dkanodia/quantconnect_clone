"""
Shared utilities for all DataFeed implementations.

Nothing in here is part of the public API — feeds import from this module
internally. All types imported from backtester.interfaces and
backtester.exceptions only (no cross-sibling imports).
"""

from __future__ import annotations

from datetime import datetime, timezone

import polars as pl

from backtester.exceptions import DataCorruptionError
from backtester.interfaces import Bar

# Columns that every OHLCV feed must provide (lowercase).
REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {"timestamp", "open", "high", "low", "close", "volume"}
)


def validate_columns(df: pl.DataFrame, source: str) -> None:
    """
    Raise DataCorruptionError if any required OHLCV column is absent.

    Comparison is case-insensitive; column names must already be lowercased
    before this function is called.
    """
    present = set(df.columns)
    missing = REQUIRED_COLUMNS - present
    if missing:
        raise DataCorruptionError(
            f"Missing required columns {sorted(missing)} in '{source}'. "
            f"Found: {sorted(df.columns)}."
        )


def normalize_timestamp_series(series: pl.Series) -> pl.Series:
    """
    Coerce a timestamp column (any supported dtype) to Datetime[μs, UTC].

    Supported input dtypes
    ----------------------
    Datetime (any time_unit / time_zone)
        Already a datetime — timezone is pinned or converted to UTC.
    Date
        Cast to microsecond Datetime, then marked UTC.
    Int8 / Int16 / Int32 / Int64 / UInt8 / UInt16 / UInt32 / UInt64
        Treated as Unix epoch **seconds** → scaled to microseconds then cast.
    String (Utf8)
        Parsed with polars' built-in inference (ISO 8601, YYYY-MM-DD, etc.).

    Raises
    ------
    DataCorruptionError
        If the series cannot be coerced to a datetime representation.
    """
    dtype_name = type(series.dtype).__name__

    if dtype_name == "Datetime":
        tz: str | None = getattr(series.dtype, "time_zone", None)
        if tz is None:
            return series.dt.replace_time_zone("UTC")
        if tz != "UTC":
            return series.dt.convert_time_zone("UTC")
        return series  # Already Datetime[*, UTC]

    if dtype_name == "Date":
        return series.cast(pl.Datetime("us")).dt.replace_time_zone("UTC")

    if dtype_name in (
        "Int8", "Int16", "Int32", "Int64",
        "UInt8", "UInt16", "UInt32", "UInt64",
    ):
        # Scale seconds → microseconds, then cast to Datetime and mark UTC.
        return (
            (series.cast(pl.Int64) * 1_000_000)
            .cast(pl.Datetime("us"))
            .dt.replace_time_zone("UTC")
        )

    if dtype_name in ("Utf8", "String"):
        try:
            parsed = series.str.to_datetime(strict=False, time_unit="us")
            return parsed.dt.replace_time_zone("UTC")
        except Exception as exc:
            raise DataCorruptionError(
                f"Cannot parse timestamp strings. "
                f"Sample values: {series[:3].to_list()}. Error: {exc}"
            ) from exc

    raise DataCorruptionError(
        f"Cannot convert timestamp dtype {dtype_name!r} to datetime. "
        "Expected Datetime, Date, integer (Unix epoch seconds), or string."
    )


def polars_df_to_bars(df: pl.DataFrame, symbol: str, source: str) -> list[Bar]:
    """
    Convert a Polars DataFrame to a list of Bar objects sorted by timestamp.

    The DataFrame must contain these columns (case-insensitive, no symbol col):
        timestamp, open, high, low, close, volume

    Parameters
    ----------
    df:     DataFrame without a ``symbol`` column (caller must drop/filter it).
    symbol: Ticker string assigned to every Bar produced.
    source: Human-readable label used in error messages (e.g. file path).

    Returns
    -------
    list[Bar]
        Bars sorted ascending by timestamp.

    Raises
    ------
    DataCorruptionError
        On missing columns or unparseable timestamps.
    """
    # Normalise column names to lowercase (idempotent if already done).
    rename_map = {c: c.lower() for c in df.columns if c != c.lower()}
    if rename_map:
        df = df.rename(rename_map)

    validate_columns(df, source)

    try:
        ts_col = normalize_timestamp_series(df["timestamp"])
    except DataCorruptionError:
        raise
    except Exception as exc:
        raise DataCorruptionError(
            f"Timestamp normalisation failed in '{source}': {exc}"
        ) from exc

    df = df.with_columns(ts_col.alias("timestamp")).sort("timestamp")

    bars: list[Bar] = []
    for row in df.iter_rows(named=True):
        ts = row["timestamp"]
        if isinstance(ts, datetime) and ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        bars.append(
            Bar(
                symbol=symbol,
                timestamp=ts,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )
        )

    return bars
