"""
models/trend_regime_model.py

Trend + volatility regime model for ETH.

This model answers:
- Is the market bullish / bearish / neutral?
- Is volatility expanding (good for long options)?
- Is the regime hostile to long options?

No predictions, only regime classification.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd

from data.market_data import download_eth_data, get_close_prices


@dataclass
class TrendRegime:
    bullish: bool
    bearish: bool
    volatility_expanding: bool
    hostile: bool
    sma_fast: float
    sma_slow: float
    hv_fast: float
    hv_slow: float


def annualized_vol(prices: pd.Series, window: int) -> float:
    returns = np.log(prices / prices.shift(1)).dropna()
    if len(returns) < window:
        return 0.0
    return float(returns.tail(window).std(ddof=1) * np.sqrt(365))


def get_trend_regime(
    start_date: str = "2023-01-01",
    sma_fast_window: int = 20,
    sma_slow_window: int = 50,
    hv_fast_window: int = 20,
    hv_slow_window: int = 60,
) -> TrendRegime:
    data = download_eth_data(start_date=start_date)
    close = get_close_prices(data).dropna()

    if len(close) < max(sma_slow_window, hv_slow_window) + 5:
        raise RuntimeError("Not enough data for trend regime")

    sma_fast = float(close.tail(sma_fast_window).mean())
    sma_slow = float(close.tail(sma_slow_window).mean())

    hv_fast = annualized_vol(close, hv_fast_window)
    hv_slow = annualized_vol(close, hv_slow_window)

    bullish = sma_fast > sma_slow
    bearish = sma_fast < sma_slow
    volatility_expanding = hv_fast > hv_slow

    hostile = not volatility_expanding

    return TrendRegime(
        bullish=bullish,
        bearish=bearish,
        volatility_expanding=volatility_expanding,
        hostile=hostile,
        sma_fast=sma_fast,
        sma_slow=sma_slow,
        hv_fast=hv_fast,
        hv_slow=hv_slow,
    )