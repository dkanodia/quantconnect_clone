"""
Data layer — concrete DataFeed implementations.

Public API
----------
CSVFeed      reads from a CSV file or directory of CSV files
ParquetFeed  reads from a Parquet file or directory of Parquet files
YFinanceFeed downloads from Yahoo Finance via yfinance
"""

from backtester.data.csv_feed import CSVFeed
from backtester.data.parquet_feed import ParquetFeed
from backtester.data.yfinance_feed import YFinanceFeed

__all__ = ["CSVFeed", "ParquetFeed", "YFinanceFeed"]
