from __future__ import annotations
import math
import numpy as np
import pandas as pd


def log_returns(prices: pd.Series) -> pd.Series:
    close = pd.to_numeric(prices, errors='coerce').dropna()
    close = close[close > 0]
    return np.log(close / close.shift(1)).dropna()


def simple_returns(prices: pd.Series) -> pd.Series:
    return pd.to_numeric(prices, errors='coerce').dropna().pct_change().dropna()


def annualization_factor(timeframe: str = '1d') -> int:
    return {'1d': 365, '1h': 365 * 24, '4h': 365 * 6}.get(str(timeframe).lower(), 365)


def historical_volatility(prices: pd.Series, timeframe: str = '1d', use_log_returns: bool = True) -> float:
    returns = log_returns(prices) if use_log_returns else simple_returns(prices)
    if len(returns) < 2:
        return 0.0
    return float(returns.std(ddof=1) * math.sqrt(annualization_factor(timeframe)))


def rolling_volatility(prices: pd.Series, window: int = 30, timeframe: str = '1d', use_log_returns: bool = True) -> pd.Series:
    returns = log_returns(prices) if use_log_returns else simple_returns(prices)
    return returns.rolling(window).std(ddof=1) * math.sqrt(annualization_factor(timeframe))


def realized_volatility(prices: pd.Series, window: int = 30, timeframe: str = '1d') -> pd.Series:
    return rolling_volatility(prices, window, timeframe)


def volatility_rank(current_volatility: float, volatility_series: pd.Series) -> float:
    series = pd.to_numeric(volatility_series, errors='coerce').dropna()
    if series.empty or series.max() <= series.min():
        return 0.5
    return float(max(0.0, min(1.0, (current_volatility - series.min()) / (series.max() - series.min()))))


def volatility_zscore(volatility_series: pd.Series, window: int = 100) -> pd.Series:
    series = pd.to_numeric(volatility_series, errors='coerce')
    return (series - series.rolling(window).mean()) / series.rolling(window).std(ddof=1)


def summarize_volatility(prices: pd.Series, timeframe: str = '1d', rolling_window: int = 30) -> dict[str, float]:
    return {'historical_volatility': historical_volatility(prices, timeframe)}
