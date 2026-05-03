"""
models/volatility.py

Volatility models and helper functions for ETH options.

This module calculates:
- log returns
- simple returns
- historical volatility
- rolling volatility
- realized volatility
- annualized volatility
- volatility summary statistics

Volatility is one of the most important inputs in Black-Scholes.
For ETH options, volatility can change quickly, so using a fixed volatility
assumption is usually not enough for serious research.

Important:
This module is for research and education. It is not financial advice.
"""

import numpy as np
import pandas as pd


def simple_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate simple percentage returns.

    Simple return:
        R_t = P_t / P_{t-1} - 1

    Parameters
    ----------
    prices : pd.Series
        Series of asset prices.

    Returns
    -------
    pd.Series
        Simple returns.
    """

    if prices.empty:
        raise ValueError("prices cannot be empty.")

    if (prices <= 0).any():
        raise ValueError("prices must be greater than 0.")

    return prices.pct_change()


def log_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate logarithmic returns.

    Log return:
        r_t = ln(P_t / P_{t-1})

    Log returns are commonly used for volatility estimation.

    Parameters
    ----------
    prices : pd.Series
        Series of asset prices.

    Returns
    -------
    pd.Series
        Log returns.
    """

    if prices.empty:
        raise ValueError("prices cannot be empty.")

    if (prices <= 0).any():
        raise ValueError("prices must be greater than 0.")

    ratio = prices / prices.shift(1)
    return pd.Series(np.log(ratio), index=ratio.index, name=prices.name)


def annualization_factor(timeframe: str) -> int:
    """
    Return annualization factor for a given timeframe.

    Crypto trades 24/7, so common assumptions are:
    - daily data: 365 periods per year
    - hourly data: 365 * 24 periods per year
    - 4h data: 365 * 6 periods per year
    - 15m data: 365 * 24 * 4 periods per year
    - 5m data: 365 * 24 * 12 periods per year
    - 1m data: 365 * 24 * 60 periods per year

    Parameters
    ----------
    timeframe : str
        Data timeframe, for example "1d", "1h", "4h", "15m", "5m", "1m".

    Returns
    -------
    int
        Number of periods per year.
    """

    timeframe = timeframe.lower().strip()

    mapping = {
        "1d": 365,
        "1day": 365,
        "daily": 365,
        "1h": 365 * 24,
        "hourly": 365 * 24,
        "4h": 365 * 6,
        "15m": 365 * 24 * 4,
        "5m": 365 * 24 * 12,
        "1m": 365 * 24 * 60,
    }

    if timeframe not in mapping:
        raise ValueError(
            "Unsupported timeframe. Use one of: "
            "'1d', '1h', '4h', '15m', '5m', '1m'."
        )

    return mapping[timeframe]


def historical_volatility(
    prices: pd.Series,
    timeframe: str = "1d",
    use_log_returns: bool = True,
) -> float:
    """
    Calculate annualized historical volatility over the full price series.

    Parameters
    ----------
    prices : pd.Series
        Series of ETH prices.
    timeframe : str
        Data timeframe.
    use_log_returns : bool
        If True, use log returns. If False, use simple returns.

    Returns
    -------
    float
        Annualized volatility as decimal.

    Example:
        0.75 means 75% annualized volatility.
    """

    factor = annualization_factor(timeframe)

    returns = log_returns(prices) if use_log_returns else simple_returns(prices)
    returns = returns.dropna()

    if returns.empty:
        raise ValueError("not enough price data to calculate volatility.")

    volatility = returns.std(ddof=1) * np.sqrt(factor)

    return float(volatility)


def rolling_volatility(
    prices: pd.Series,
    window: int = 30,
    timeframe: str = "1d",
    use_log_returns: bool = True,
) -> pd.Series:
    """
    Calculate rolling annualized volatility.

    Parameters
    ----------
    prices : pd.Series
        Series of ETH prices.
    window : int
        Rolling window length.
        Example: 30 for daily data means 30-day rolling volatility.
    timeframe : str
        Data timeframe.
    use_log_returns : bool
        If True, use log returns. If False, use simple returns.

    Returns
    -------
    pd.Series
        Rolling annualized volatility.
    """

    if window <= 1:
        raise ValueError("window must be greater than 1.")

    factor = annualization_factor(timeframe)

    returns = log_returns(prices) if use_log_returns else simple_returns(prices)

    vol = returns.rolling(window=window).std(ddof=1) * np.sqrt(factor)

    return vol


def realized_volatility(
    prices: pd.Series,
    window: int,
    timeframe: str = "1h",
) -> pd.Series:
    """
    Calculate rolling realized volatility.

    This is especially useful for intraday ETH data.

    Parameters
    ----------
    prices : pd.Series
        Series of ETH prices.
    window : int
        Rolling window.
        Example: for hourly data, window=24 gives 24-hour realized volatility.
    timeframe : str
        Data timeframe.

    Returns
    -------
    pd.Series
        Rolling realized volatility.
    """

    return rolling_volatility(
        prices=prices,
        window=window,
        timeframe=timeframe,
        use_log_returns=True,
    )


def volatility_rank(
    current_volatility: float,
    volatility_series: pd.Series,
) -> float:
    """
    Calculate volatility rank.

    Volatility rank shows where current volatility sits compared to
    historical volatility values.

    Formula:
        rank = percentage of historical volatility values below current volatility

    Returns
    -------
    float
        Volatility rank between 0 and 1.

    Example:
        0.80 means current volatility is higher than 80% of historical observations.
    """

    clean_vol = volatility_series.dropna()

    if clean_vol.empty:
        raise ValueError("volatility_series does not contain valid values.")

    rank = (clean_vol < current_volatility).mean()

    return float(rank)


def volatility_zscore(
    volatility_series: pd.Series,
    window: int = 100,
) -> pd.Series:
    """
    Calculate rolling z-score of volatility.

    Z-score:
        z = (current volatility - rolling mean) / rolling std

    This helps detect unusually high or low volatility regimes.
    """

    if window <= 1:
        raise ValueError("window must be greater than 1.")

    rolling_mean = volatility_series.rolling(window=window).mean()
    rolling_std = volatility_series.rolling(window=window).std(ddof=1)

    zscore = (volatility_series - rolling_mean) / rolling_std

    return zscore


def summarize_volatility(
    prices: pd.Series,
    timeframe: str = "1d",
    rolling_window: int = 30,
) -> dict[str, float]:
    """
    Create a volatility summary.

    Returns:
    - full-period historical volatility
    - latest rolling volatility
    - mean rolling volatility
    - median rolling volatility
    - min rolling volatility
    - max rolling volatility
    - latest volatility rank
    """

    hist_vol = historical_volatility(
        prices=prices,
        timeframe=timeframe,
        use_log_returns=True,
    )

    rolling_vol = rolling_volatility(
        prices=prices,
        window=rolling_window,
        timeframe=timeframe,
        use_log_returns=True,
    )

    clean_rolling = rolling_vol.dropna()

    if clean_rolling.empty:
        raise ValueError("not enough data for rolling volatility summary.")

    latest_vol = float(clean_rolling.iloc[-1])

    return {
        "historical_volatility": float(hist_vol),
        "latest_rolling_volatility": latest_vol,
        "mean_rolling_volatility": float(clean_rolling.mean()),
        "median_rolling_volatility": float(clean_rolling.median()),
        "min_rolling_volatility": float(clean_rolling.min()),
        "max_rolling_volatility": float(clean_rolling.max()),
        "latest_volatility_rank": volatility_rank(latest_vol, clean_rolling),
    }


if __name__ == "__main__":
    # Simple standalone test using fake ETH price data.

    np.random.seed(42)

    # Create synthetic ETH-like prices for testing only.
    # This is NOT real market data.
    days = 365
    starting_price = 3000.0

    simulated_daily_returns = np.random.normal(
        loc=0.0005,
        scale=0.04,
        size=days,
    )

    simulated_prices = starting_price * np.exp(
        np.cumsum(simulated_daily_returns)
    )

    prices = pd.Series(simulated_prices)

    summary = summarize_volatility(
        prices=prices,
        timeframe="1d",
        rolling_window=30,
    )

    print("========== ETH Volatility Summary Test ==========")
    for name, value in summary.items():
        print(f"{name:<30}: {value:.4f}")
    print("================================================")