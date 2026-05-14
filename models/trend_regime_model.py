from __future__ import annotations
from dataclasses import dataclass
import numpy as np
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


def annualized_vol(prices, window: int) -> float:
    returns = np.log(prices / prices.shift(1)).dropna()
    return 0.0 if len(returns) < window else float(returns.tail(window).std(ddof=1) * np.sqrt(365))


def get_trend_regime(start_date: str = '2023-01-01', sma_fast_window: int = 20, sma_slow_window: int = 50, hv_fast_window: int = 20, hv_slow_window: int = 60) -> TrendRegime:
    try:
        close = get_close_prices(download_eth_data(start_date=start_date)).dropna()
    except Exception:
        return TrendRegime(False, False, False, False, 0, 0, 0, 0)
    if len(close) < 60:
        return TrendRegime(False, False, False, False, float(close.iloc[-1]) if len(close) else 0, 0, 0, 0)
    sma_fast = float(close.tail(sma_fast_window).mean())
    sma_slow = float(close.tail(sma_slow_window).mean())
    hv_fast = annualized_vol(close, hv_fast_window)
    hv_slow = annualized_vol(close, hv_slow_window)
    return TrendRegime(sma_fast > sma_slow * 1.005, sma_fast < sma_slow * 0.995, hv_fast > hv_slow * 1.05, False, sma_fast, sma_slow, hv_fast, hv_slow)
