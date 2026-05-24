"""
Yahoo Finance DataFeed implementation via yfinance.

Downloaded data is cached in memory at construction time so that reset()
and repeated iteration never trigger additional network requests.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timezone
from typing import Union
from typing import Iterator

import pandas as pd

from backtester.exceptions import DataFeedError
from backtester.interfaces import Bar, DataFeed


class YFinanceFeed(DataFeed):
    """
    DataFeed that downloads OHLCV data from Yahoo Finance via ``yfinance``.

    Each ticker is fetched individually so that a failed ticker raises an
    error early and cannot silently corrupt a merged MultiIndex DataFrame.
    All downloaded data is cached in memory; ``reset()`` and repeated
    iteration never trigger a second download.

    Parameters
    ----------
    tickers:
        List of ticker symbols.  Symbols are upper-cased automatically.
    start:
        Start date as an ISO 8601 string (``"YYYY-MM-DD"``), a
        ``datetime.date``, or a ``datetime.datetime``.
    end:
        End date in the same formats as *start*.
    interval:
        Bar size accepted by yfinance.  Defaults to ``"1d"`` (daily).
        Common values: ``"1m"``, ``"5m"``, ``"15m"``, ``"1h"``,
        ``"1d"``, ``"1wk"``, ``"1mo"``.

    Raises
    ------
    DataFeedError
        If yfinance is not installed, a download returns empty data, or
        the network request fails.
    """

    def __init__(
        self,
        tickers: list[str],
        start: Union[str, date, datetime],
        end: Union[str, date, datetime],
        interval: str = "1d",
    ) -> None:
        self._tickers: list[str] = [t.upper() for t in tickers]
        self._start = start
        self._end = end
        self._interval = interval
        # _bars is populated once at construction; never re-downloaded.
        self._bars: list[Bar] = self._download_all()

    # ------------------------------------------------------------------
    # DataFeed interface
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[Bar]:
        """Yield all bars in ascending timestamp order."""
        return iter(self._bars)

    def reset(self) -> None:
        """
        Reset the feed to its initial state.

        The download cache is retained; each call to ``__iter__`` already
        returns a fresh iterator from the first bar.  This is a no-op.
        """

    @property
    def symbols(self) -> list[str]:
        """Return the tickers supplied at construction (in upper-case)."""
        return list(self._tickers)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _download_all(self) -> list[Bar]:
        """Download every ticker sequentially and return merged sorted bars."""
        try:
            import yfinance as yf  # noqa: PLC0415  (intentional lazy import)
        except ImportError as exc:
            raise DataFeedError(
                "yfinance is not installed. Install it with: pip install yfinance"
            ) from exc

        all_bars: list[Bar] = []
        for ticker in self._tickers:
            all_bars.extend(self._download_one(yf, ticker))

        return sorted(all_bars, key=lambda b: b.timestamp)

    def _download_one(self, yf: object, ticker: str) -> list[Bar]:
        """
        Download a single ticker and return its Bar list.

        Parameters
        ----------
        yf:     The imported yfinance module object.
        ticker: Upper-cased ticker symbol.
        """
        try:
            raw: pd.DataFrame = yf.download(  # type: ignore[attr-defined]
                ticker,
                start=self._start,
                end=self._end,
                interval=self._interval,
                auto_adjust=False,
                progress=False,
                multi_level_index=False,
            )
        except Exception as exc:
            raise DataFeedError(
                f"yfinance download failed for '{ticker}' "
                f"(start={self._start}, end={self._end}, "
                f"interval={self._interval}): {exc}"
            ) from exc

        if raw is None or raw.empty:
            raise DataFeedError(
                f"No data returned by yfinance for '{ticker}' "
                f"(start={self._start}, end={self._end}, "
                f"interval={self._interval}). "
                "Check that the ticker is valid and the date range is non-empty."
            )

        return self._df_to_bars(raw, ticker)

    def _df_to_bars(self, df: pd.DataFrame, symbol: str) -> list[Bar]:
        """
        Convert a single-ticker yfinance DataFrame to a list of Bar objects.

        yfinance may return a MultiIndex DataFrame even for single tickers
        depending on the version and ``multi_level_index`` flag.  This method
        handles both flat and MultiIndex column layouts.

        NaN price rows are silently skipped; NaN volume is replaced with 0.
        """
        # Flatten MultiIndex columns if present (level-0 = field, level-1 = ticker).
        if isinstance(df.columns, pd.MultiIndex):
            # Try to extract the ticker slice; fall back to level-0 fields only.
            try:
                df = df.xs(symbol, axis=1, level=1, drop_level=True)
            except KeyError:
                df.columns = df.columns.get_level_values(0)

        # Prefer unadjusted 'Close'; fall back to 'Adj Close' if unavailable.
        close_col = "Close" if "Close" in df.columns else "Adj Close"
        if close_col not in df.columns:
            raise DataFeedError(
                f"Cannot locate a 'Close' or 'Adj Close' column for '{symbol}'. "
                f"Available columns: {list(df.columns)}"
            )

        bars: list[Bar] = []
        for ts, row in df.iterrows():
            try:
                o = float(row["Open"])
                h = float(row["High"])
                lo = float(row["Low"])
                c = float(row[close_col])
                vol_raw = row.get("Volume", 0.0)
                vol = 0.0 if (vol_raw != vol_raw) else float(vol_raw)  # NaN → 0

                # Skip rows where any price is NaN.
                if any(math.isnan(v) for v in (o, h, lo, c)):
                    continue

                # Convert pandas Timestamp to timezone-aware Python datetime.
                if isinstance(ts, pd.Timestamp):
                    ts_dt = ts.to_pydatetime()
                else:
                    ts_dt = datetime(ts.year, ts.month, ts.day)

                if ts_dt.tzinfo is None:
                    ts_dt = ts_dt.replace(tzinfo=timezone.utc)

                bars.append(
                    Bar(
                        symbol=symbol,
                        timestamp=ts_dt,
                        open=o,
                        high=h,
                        low=lo,
                        close=c,
                        volume=vol,
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise DataFeedError(
                    f"Cannot convert row for '{symbol}' at index {ts}: {exc}"
                ) from exc

        return bars
