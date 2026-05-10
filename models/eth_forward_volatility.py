"""ETH-specific forward volatility model.

Creates a dynamic ETH forward-volatility estimate from multiple realized-volatility
horizons, EWMA volatility, vol-of-vol, and recent jump risk. This avoids relying
on one fixed historical-volatility input for option pricing and trade selection.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365.0


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
    status: str = "ok"


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def sigmoid(x: float) -> float:
    if x >= 40:
        return 1.0
    if x <= -40:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def log_returns(prices: pd.Series) -> pd.Series:
    clean = pd.to_numeric(prices, errors="coerce").dropna()
    clean = clean[clean > 0]
    if len(clean) < 2:
        return pd.Series(dtype=float)
    return np.log(clean / clean.shift(1)).dropna()


def annualized_realized_volatility(prices: pd.Series, window: int) -> float:
    returns = log_returns(prices)
    if len(returns) < max(2, window):
        return 0.0
    return float(returns.tail(window).std(ddof=1) * math.sqrt(DAYS_PER_YEAR))


def ewma_volatility(prices: pd.Series, span: int = 20) -> float:
    returns = log_returns(prices)
    if len(returns) < 5:
        return 0.0
    variance = returns.pow(2).ewm(span=span, adjust=False).mean().iloc[-1]
    return float(math.sqrt(max(float(variance), 0.0) * DAYS_PER_YEAR))


def rolling_vol_series(prices: pd.Series, window: int = 14) -> pd.Series:
    returns = log_returns(prices)
    if len(returns) < window:
        return pd.Series(dtype=float)
    return returns.rolling(window).std(ddof=1).dropna() * math.sqrt(DAYS_PER_YEAR)


def volatility_rank(current_vol: float, vol_series: pd.Series) -> float:
    series = pd.to_numeric(vol_series, errors="coerce").dropna()
    current = safe_float(current_vol)
    if series.empty or current <= 0:
        return 0.5
    low, high = float(series.min()), float(series.max())
    if high <= low:
        return 0.5
    return clamp((current - low) / (high - low), 0.0, 1.0)


def volatility_zscore(current_vol: float, vol_series: pd.Series) -> float:
    series = pd.to_numeric(vol_series, errors="coerce").dropna()
    current = safe_float(current_vol)
    if len(series) < 5:
        return 0.0
    mean = float(series.mean())
    std = float(series.std(ddof=1))
    if std <= 0:
        return 0.0
    return float((current - mean) / std)


def jump_risk_from_returns(prices: pd.Series, lookback: int = 30) -> float:
    returns = log_returns(prices)
    if len(returns) < 10:
        return 0.5
    recent = returns.tail(lookback).abs()
    large_move_fraction = float((recent > 0.06).mean())
    max_move = float(recent.max()) if not recent.empty else 0.0
    score = 0.65 * clamp(large_move_fraction / 0.25, 0.0, 1.0) + 0.35 * clamp(max_move / 0.15, 0.0, 1.0)
    return clamp(score, 0.0, 1.0)


def calculate_volatility_forecast(prices: pd.Series) -> VolatilityForecast:
    clean = pd.to_numeric(prices, errors="coerce").dropna()
    clean = clean[clean > 0]
    if len(clean) < 35:
        return VolatilityForecast(
            forecast_vol=0.0,
            rv_7d=0.0,
            rv_14d=0.0,
            rv_30d=0.0,
            ewma_vol=0.0,
            vol_of_vol=0.0,
            vol_rank=0.5,
            vol_zscore=0.0,
            expansion_score=0.5,
            contraction_score=0.5,
            jump_risk_score=0.5,
            status="not_enough_price_history",
        )

    rv_7d = annualized_realized_volatility(clean, 7)
    rv_14d = annualized_realized_volatility(clean, 14)
    rv_30d = annualized_realized_volatility(clean, 30)
    ewma = ewma_volatility(clean, span=20)
    vol_series = rolling_vol_series(clean, 14)

    vol_of_vol = float(vol_series.tail(30).std(ddof=1)) if len(vol_series) >= 10 else 0.0
    current_ref = rv_14d if rv_14d > 0 else rv_30d
    rank = volatility_rank(current_ref, vol_series)
    z = volatility_zscore(current_ref, vol_series)

    jump_score = jump_risk_from_returns(clean)
    fast_slow_ratio = rv_7d / rv_30d if rv_30d > 0 else 1.0
    expansion_score = clamp(0.65 * sigmoid((fast_slow_ratio - 1.0) / 0.15) + 0.35 * sigmoid(z / 1.25), 0.0, 1.0)
    contraction_score = clamp(1.0 - expansion_score, 0.0, 1.0)

    forecast = (
        0.30 * rv_7d
        + 0.25 * rv_14d
        + 0.20 * rv_30d
        + 0.15 * ewma
        + 0.05 * vol_of_vol
        + 0.05 * jump_score * max(rv_7d, rv_14d, rv_30d, ewma)
    )

    return VolatilityForecast(
        forecast_vol=float(clamp(forecast, 0.05, 4.00)),
        rv_7d=float(rv_7d),
        rv_14d=float(rv_14d),
        rv_30d=float(rv_30d),
        ewma_vol=float(ewma),
        vol_of_vol=float(max(vol_of_vol, 0.0)),
        vol_rank=float(rank),
        vol_zscore=float(z),
        expansion_score=float(expansion_score),
        contraction_score=float(contraction_score),
        jump_risk_score=float(jump_score),
        status="ok",
    )
