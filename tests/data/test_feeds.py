"""
Phase 2 tests — data layer: ParquetFeed, CSVFeed, YFinanceFeed.

Each feed is tested with a minimal in-memory or file-system fixture.
YFinanceFeed tests mock yfinance.download so no network calls are made.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import polars as pl
import pytest

from backtester.data import CSVFeed, ParquetFeed, YFinanceFeed
from backtester.exceptions import DataCorruptionError, DataFeedError, DataNotFoundError
from backtester.interfaces import Bar


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------

_UTC = timezone.utc

_DATES = [
    datetime(2024, 1, 2, tzinfo=_UTC),
    datetime(2024, 1, 3, tzinfo=_UTC),
    datetime(2024, 1, 4, tzinfo=_UTC),
]


def _ohlcv_df(symbol: str | None = None) -> pl.DataFrame:
    """Return a minimal 3-row OHLCV Polars DataFrame."""
    data: dict = {
        "timestamp": _DATES,
        "open":      [100.0, 101.0, 102.0],
        "high":      [105.0, 106.0, 107.0],
        "low":       [98.0,  99.0,  100.0],
        "close":     [103.0, 104.0, 105.0],
        "volume":    [1_000_000.0, 1_100_000.0, 1_200_000.0],
    }
    if symbol is not None:
        data["symbol"] = [symbol] * 3
    return pl.DataFrame(data)


def _yf_df(ticker: str) -> pd.DataFrame:
    """Return a mock yfinance single-ticker DataFrame (flat columns)."""
    index = pd.DatetimeIndex(_DATES, name="Date")
    return pd.DataFrame(
        {
            "Open":      [100.0, 101.0, 102.0],
            "High":      [105.0, 106.0, 107.0],
            "Low":       [98.0,  99.0,  100.0],
            "Close":     [103.0, 104.0, 105.0],
            "Adj Close": [103.0, 104.0, 105.0],
            "Volume":    [1_000_000, 1_100_000, 1_200_000],
        },
        index=index,
    )


# ---------------------------------------------------------------------------
# ParquetFeed
# ---------------------------------------------------------------------------


class TestParquetFeed:
    def test_single_symbol_file_infers_symbol_from_filename(
        self, tmp_path: Path
    ) -> None:
        fpath = tmp_path / "SPY.parquet"
        _ohlcv_df().write_parquet(str(fpath))

        feed = ParquetFeed(fpath)
        bars = list(feed)

        assert len(bars) == 3
        assert all(b.symbol == "SPY" for b in bars)

    def test_bars_ascending_timestamp_order(self, tmp_path: Path) -> None:
        fpath = tmp_path / "AAPL.parquet"
        _ohlcv_df().write_parquet(str(fpath))

        bars = list(ParquetFeed(fpath))
        timestamps = [b.timestamp for b in bars]

        assert timestamps == sorted(timestamps)

    def test_bar_ohlcv_values(self, tmp_path: Path) -> None:
        fpath = tmp_path / "SPY.parquet"
        _ohlcv_df().write_parquet(str(fpath))

        bar = list(ParquetFeed(fpath))[0]

        assert bar.open   == pytest.approx(100.0)
        assert bar.high   == pytest.approx(105.0)
        assert bar.low    == pytest.approx(98.0)
        assert bar.close  == pytest.approx(103.0)
        assert bar.volume == pytest.approx(1_000_000.0)

    def test_multi_symbol_file_via_symbol_column(self, tmp_path: Path) -> None:
        rows = {
            "timestamp": _DATES[:2] + _DATES[:2],
            "symbol":    ["SPY", "SPY", "QQQ", "QQQ"],
            "open":      [100.0, 101.0, 200.0, 201.0],
            "high":      [105.0, 106.0, 205.0, 206.0],
            "low":       [98.0,  99.0,  198.0, 199.0],
            "close":     [103.0, 104.0, 203.0, 204.0],
            "volume":    [1e6,   1.1e6, 2e6,   2.1e6],
        }
        fpath = tmp_path / "multi.parquet"
        pl.DataFrame(rows).write_parquet(str(fpath))

        feed = ParquetFeed(fpath)

        assert set(feed.symbols) == {"SPY", "QQQ"}
        assert len(list(feed)) == 4

    def test_symbols_property_sorted(self, tmp_path: Path) -> None:
        fpath = tmp_path / "TSLA.parquet"
        _ohlcv_df().write_parquet(str(fpath))

        assert ParquetFeed(fpath).symbols == ["TSLA"]

    def test_directory_merges_multiple_files(self, tmp_path: Path) -> None:
        _ohlcv_df().write_parquet(str(tmp_path / "SPY.parquet"))
        _ohlcv_df().write_parquet(str(tmp_path / "QQQ.parquet"))

        feed = ParquetFeed(tmp_path)

        assert set(feed.symbols) == {"SPY", "QQQ"}
        assert len(list(feed)) == 6

    def test_directory_bars_globally_sorted(self, tmp_path: Path) -> None:
        _ohlcv_df().write_parquet(str(tmp_path / "SPY.parquet"))
        _ohlcv_df().write_parquet(str(tmp_path / "QQQ.parquet"))

        bars = list(ParquetFeed(tmp_path))
        timestamps = [b.timestamp for b in bars]

        assert timestamps == sorted(timestamps)

    def test_reset_allows_reiteration(self, tmp_path: Path) -> None:
        fpath = tmp_path / "SPY.parquet"
        _ohlcv_df().write_parquet(str(fpath))
        feed = ParquetFeed(fpath)

        first  = list(feed)
        feed.reset()
        second = list(feed)

        assert [b.timestamp for b in first] == [b.timestamp for b in second]

    def test_missing_required_column_raises(self, tmp_path: Path) -> None:
        fpath = tmp_path / "bad.parquet"
        pl.DataFrame(
            {"timestamp": _DATES[:1], "open": [100.0]}  # missing high/low/close/volume
        ).write_parquet(str(fpath))

        with pytest.raises(DataCorruptionError):
            ParquetFeed(fpath)

    def test_nonexistent_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DataNotFoundError):
            ParquetFeed(tmp_path / "ghost.parquet")

    def test_empty_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DataNotFoundError):
            ParquetFeed(tmp_path)

    def test_uppercase_column_names_normalised(self, tmp_path: Path) -> None:
        """Feeds must accept UPPER or Mixed case column names."""
        fpath = tmp_path / "SPY.parquet"
        pl.DataFrame(
            {
                "Timestamp": _DATES,
                "Open":      [100.0, 101.0, 102.0],
                "High":      [105.0, 106.0, 107.0],
                "Low":       [98.0,  99.0,  100.0],
                "Close":     [103.0, 104.0, 105.0],
                "Volume":    [1e6,   1.1e6, 1.2e6],
            }
        ).write_parquet(str(fpath))

        bars = list(ParquetFeed(fpath))
        assert len(bars) == 3


# ---------------------------------------------------------------------------
# CSVFeed
# ---------------------------------------------------------------------------

_CSV_SINGLE = """\
timestamp,open,high,low,close,volume
2024-01-02,100.0,105.0,98.0,103.0,1000000
2024-01-03,101.0,106.0,99.0,104.0,1100000
2024-01-04,102.0,107.0,100.0,105.0,1200000
"""

_CSV_MULTI = """\
timestamp,symbol,open,high,low,close,volume
2024-01-02,SPY,100.0,105.0,98.0,103.0,1000000
2024-01-02,QQQ,200.0,205.0,198.0,203.0,2000000
2024-01-03,SPY,101.0,106.0,99.0,104.0,1100000
2024-01-03,QQQ,201.0,206.0,199.0,204.0,2100000
"""

_CSV_EPOCH = """\
timestamp,open,high,low,close,volume
1704153600,100.0,105.0,98.0,103.0,1000000
"""


class TestCSVFeed:
    def test_single_symbol_file_infers_symbol_from_filename(
        self, tmp_path: Path
    ) -> None:
        fpath = tmp_path / "SPY.csv"
        fpath.write_text(_CSV_SINGLE)

        feed = CSVFeed(fpath)
        bars = list(feed)

        assert len(bars) == 3
        assert all(b.symbol == "SPY" for b in bars)

    def test_bars_ascending_timestamp_order(self, tmp_path: Path) -> None:
        fpath = tmp_path / "SPY.csv"
        fpath.write_text(_CSV_SINGLE)

        bars = list(CSVFeed(fpath))
        timestamps = [b.timestamp for b in bars]

        assert timestamps == sorted(timestamps)

    def test_bar_ohlcv_values(self, tmp_path: Path) -> None:
        fpath = tmp_path / "SPY.csv"
        fpath.write_text(_CSV_SINGLE)

        bar = list(CSVFeed(fpath))[0]

        assert bar.open   == pytest.approx(100.0)
        assert bar.high   == pytest.approx(105.0)
        assert bar.low    == pytest.approx(98.0)
        assert bar.close  == pytest.approx(103.0)
        assert bar.volume == pytest.approx(1_000_000.0)

    def test_multi_symbol_csv(self, tmp_path: Path) -> None:
        fpath = tmp_path / "multi.csv"
        fpath.write_text(_CSV_MULTI)

        feed = CSVFeed(fpath)

        assert set(feed.symbols) == {"SPY", "QQQ"}
        assert len(list(feed)) == 4

    def test_directory_of_csvs(self, tmp_path: Path) -> None:
        (tmp_path / "SPY.csv").write_text(_CSV_SINGLE)
        (tmp_path / "QQQ.csv").write_text(_CSV_SINGLE)

        feed = CSVFeed(tmp_path)

        assert set(feed.symbols) == {"SPY", "QQQ"}
        assert len(list(feed)) == 6

    def test_symbols_property(self, tmp_path: Path) -> None:
        fpath = tmp_path / "AAPL.csv"
        fpath.write_text(_CSV_SINGLE)

        assert CSVFeed(fpath).symbols == ["AAPL"]

    def test_reset_allows_reiteration(self, tmp_path: Path) -> None:
        fpath = tmp_path / "SPY.csv"
        fpath.write_text(_CSV_SINGLE)
        feed = CSVFeed(fpath)

        first = list(feed)
        feed.reset()
        second = list(feed)

        assert len(first) == len(second)

    def test_missing_column_raises(self, tmp_path: Path) -> None:
        fpath = tmp_path / "bad.csv"
        fpath.write_text("timestamp,open\n2024-01-02,100\n")

        with pytest.raises(DataCorruptionError):
            CSVFeed(fpath)

    def test_nonexistent_path_raises(self) -> None:
        with pytest.raises(DataNotFoundError):
            CSVFeed("/nonexistent/path/file.csv")

    def test_empty_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DataNotFoundError):
            CSVFeed(tmp_path)

    def test_unix_epoch_integer_timestamps(self, tmp_path: Path) -> None:
        """Integer Unix seconds in the timestamp column must parse correctly."""
        fpath = tmp_path / "SPY.csv"
        fpath.write_text(_CSV_EPOCH)

        bars = list(CSVFeed(fpath))

        assert len(bars) == 1
        assert bars[0].open == pytest.approx(100.0)
        # 1704153600 == 2024-01-02 00:00:00 UTC
        assert bars[0].timestamp.year == 2024
        assert bars[0].timestamp.month == 1
        assert bars[0].timestamp.day == 2

    def test_uppercase_column_names_normalised(self, tmp_path: Path) -> None:
        content = "Timestamp,Open,High,Low,Close,Volume\n2024-01-02,100,105,98,103,1000000\n"
        fpath = tmp_path / "SPY.csv"
        fpath.write_text(content)

        bars = list(CSVFeed(fpath))
        assert len(bars) == 1


# ---------------------------------------------------------------------------
# YFinanceFeed
# ---------------------------------------------------------------------------


class TestYFinanceFeed:
    def test_basic_iteration(self) -> None:
        with patch("yfinance.download", return_value=_yf_df("SPY")) as mock_dl:
            feed = YFinanceFeed(["SPY"], start="2024-01-02", end="2024-01-05")
            bars = list(feed)

        assert len(bars) == 3
        mock_dl.assert_called_once()

    def test_bar_ohlcv_values(self) -> None:
        with patch("yfinance.download", return_value=_yf_df("SPY")):
            feed = YFinanceFeed(["SPY"], start="2024-01-02", end="2024-01-05")
            bar = list(feed)[0]

        assert bar.symbol == "SPY"
        assert bar.open   == pytest.approx(100.0)
        assert bar.high   == pytest.approx(105.0)
        assert bar.low    == pytest.approx(98.0)
        assert bar.close  == pytest.approx(103.0)
        assert bar.volume == pytest.approx(1_000_000.0)

    def test_bars_ascending_timestamp_order(self) -> None:
        with patch("yfinance.download", return_value=_yf_df("SPY")):
            bars = list(YFinanceFeed(["SPY"], start="2024-01-02", end="2024-01-05"))

        timestamps = [b.timestamp for b in bars]
        assert timestamps == sorted(timestamps)

    def test_symbols_property(self) -> None:
        with patch("yfinance.download", return_value=_yf_df("SPY")):
            feed = YFinanceFeed(["SPY"], start="2024-01-02", end="2024-01-05")

        assert feed.symbols == ["SPY"]

    def test_ticker_is_uppercased(self) -> None:
        with patch("yfinance.download", return_value=_yf_df("SPY")):
            feed = YFinanceFeed(["spy"], start="2024-01-02", end="2024-01-05")

        assert feed.symbols == ["SPY"]

    def test_reset_does_not_trigger_redownload(self) -> None:
        """reset() must never make a second network call."""
        with patch("yfinance.download", return_value=_yf_df("SPY")) as mock_dl:
            feed = YFinanceFeed(["SPY"], start="2024-01-02", end="2024-01-05")
            list(feed)
            feed.reset()
            list(feed)

        # Exactly one download at construction; none after reset.
        assert mock_dl.call_count == 1

    def test_multiple_tickers_merged_and_sorted(self) -> None:
        def _side_effect(ticker, **_kw):
            return _yf_df(ticker)

        with patch("yfinance.download", side_effect=_side_effect):
            feed = YFinanceFeed(["SPY", "QQQ"], start="2024-01-02", end="2024-01-05")
            bars = list(feed)

        assert set(feed.symbols) == {"SPY", "QQQ"}
        assert len(bars) == 6
        timestamps = [b.timestamp for b in bars]
        assert timestamps == sorted(timestamps)

    def test_empty_dataframe_raises_data_feed_error(self) -> None:
        with patch("yfinance.download", return_value=pd.DataFrame()):
            with pytest.raises(DataFeedError):
                YFinanceFeed(["SPY"], start="2024-01-02", end="2024-01-05")

    def test_network_exception_raises_data_feed_error(self) -> None:
        with patch("yfinance.download", side_effect=OSError("timeout")):
            with pytest.raises(DataFeedError):
                YFinanceFeed(["SPY"], start="2024-01-02", end="2024-01-05")

    def test_nan_price_rows_are_skipped(self) -> None:
        """Rows with NaN in any price column must be silently dropped."""
        import math
        df = _yf_df("SPY").copy()
        df.loc[df.index[1], "Close"] = float("nan")  # corrupt middle row

        with patch("yfinance.download", return_value=df):
            bars = list(YFinanceFeed(["SPY"], start="2024-01-02", end="2024-01-05"))

        assert len(bars) == 2
        assert all(not math.isnan(b.close) for b in bars)

    def test_custom_interval_passed_through(self) -> None:
        with patch("yfinance.download", return_value=_yf_df("SPY")) as mock_dl:
            YFinanceFeed(["SPY"], start="2024-01-02", end="2024-01-05", interval="1h")

        _, kwargs = mock_dl.call_args
        assert kwargs.get("interval") == "1h"
