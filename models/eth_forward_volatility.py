from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VolatilityForecast:
    forecast_vol: float
    rv_7d: float
    rv_14d: float
    rv_30d: float
    ewma_vol: float
    vol_of_vol: float
    vol_rank: float
    vol_zscore: float
    expansion_score: float
    contraction_score: float
    jump_risk_score: float
    status: str = 'ok'


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def calculate_volatility_forecast(prices: pd.Series) -> VolatilityForecast:
    close = pd.to_numeric(prices, errors='coerce').dropna()
    close = close[close > 0]
    if len(close) < 35:
        return VolatilityForecast(0.75, 0, 0, 0, 0, 0, 0.5, 0, 0.5, 0.5, 0.5, 'not_enough_price_history')
    returns = np.log(close / close.shift(1)).dropna()
    def rv(window: int) -> float:
        return float(returns.tail(window).std(ddof=1) * math.sqrt(365)) if len(returns) >= window else 0.0
    rv7, rv14, rv30 = rv(7), rv(14), rv(30)
    ewma = math.sqrt(float(returns.pow(2).ewm(span=20, adjust=False).mean().iloc[-1]) * 365)
    forecast = clamp(0.30 * rv7 + 0.25 * rv14 + 0.25 * rv30 + 0.20 * ewma, 0.05, 4.0)
    expansion = clamp((rv7 / rv30 if rv30 > 0 else 1.0) - 0.5, 0.0, 1.0)
    return VolatilityForecast(forecast, rv7, rv14, rv30, ewma, 0.0, 0.5, 0.0, expansion, 1.0 - expansion, 0.5, 'ok')
