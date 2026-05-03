"""
data/market_data.py

Market data utilities for ETH.

This module downloads and prepares ETH historical price data
for volatility estimation and option pricing.

Data source:
- Yahoo Finance (ETH-USD)

This code is written to be:
- Pylance-safe
- type-safe
- robust against empty or failed downloads
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf


DEFAULT_ETH_TICKER = "ETH-USD"


def download_eth_data(
    start_date: str,
    end_date: Optional[str] = None,
    interval: str = "1d",
    ticker: str = DEFAULT_ETH_TICKER,
) -> pd.DataFrame:
    """
    Download historical ETH price data from Yahoo Finance.

    Returns
    -------
    pd.DataFrame
        ETH OHLCV market data
    """

    data = yf.download(
        tickers=ticker,
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )

    # ---- PYLANCE & SAFETY FIX ----
    if data is None:
        raise ValueError("yfinance returned None instead of a DataFrame.")

    if data.empty:
        raise ValueError(
            f"No data downloaded for ticker={ticker}, "
            f"start_date={start_date}, end_date={end_date}, interval={interval}"
        )
    # --------------------------------

    # Normalize columns (yfinance can return MultiIndex)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()

    # Standardize column names
    data.columns = [str(col).lower().replace(" ", "_") for col in data.columns]

    # Normalize timestamp column
    if "date" in data.columns:
        data = data.rename(columns={"date": "timestamp"})
    elif "datetime" in data.columns:
        data = data.rename(columns={"datetime": "timestamp"})

    required_columns = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required_columns - set(data.columns)

    if missing:
        raise ValueError(f"Downloaded data missing columns: {missing}")

    data = data[
        ["timestamp", "open", "high", "low", "close", "volume"]
    ].copy()

    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values("timestamp").reset_index(drop=True)

    return data


def save_market_data(
    data: pd.DataFrame,
    file_path: str,
) -> None:
    """
    Save market data to CSV.
    """

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data.to_csv(path, index=False)


def load_market_data(
    file_path: str,
) -> pd.DataFrame:
    """
    Load market data from CSV.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    data = pd.read_csv(path)
    data["timestamp"] = pd.to_datetime(data["timestamp"])

    return data


def get_close_prices(
    data: pd.DataFrame,
) -> pd.Series:
    """
    Extract close price series.
    """

    if "close" not in data.columns:
        raise ValueError("DataFrame must contain a 'close' column.")

    close = data["close"].dropna().astype(float)

    if close.empty:
        raise ValueError("Close price series is empty.")

    return close


def get_latest_eth_price(
    data: pd.DataFrame,
) -> float:
    """
    Get the latest ETH close price.
    """

    close = get_close_prices(data)
    return float(close.iloc[-1])


def download_and_save_eth_data(
    start_date: str,
    end_date: Optional[str] = None,
    interval: str = "1d",
    output_file: str = "data/eth_usd.csv",
) -> pd.DataFrame:
    """
    Download ETH data and save it to CSV.
    """

    data = download_eth_data(
        start_date=start_date,
        end_date=end_date,
        interval=interval,
    )

    save_market_data(data, output_file)
    return data


# ---------------------------------------------------------
# Standalone test
# ---------------------------------------------------------
if __name__ == "__main__":
    eth_data = download_and_save_eth_data(
        start_date="2023-01-01",
        end_date=None,
        interval="1d",
        output_file="data/eth_usd_daily.csv",
    )

    latest_price = get_latest_eth_price(eth_data)

    print("========== ETH MARKET DATA TEST ==========")
    print(eth_data.tail())
    print("------------------------------------------")
    print(f"Rows downloaded: {len(eth_data)}")
    print(f"Latest ETH close price: ${latest_price:,.2f}")
    print("Saved to: data/eth_usd_daily.csv")
    print("==========================================")