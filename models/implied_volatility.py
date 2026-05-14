from __future__ import annotations
from models.black_scholes import black_scholes_price, intrinsic_value


def implied_volatility(market_price: float, spot: float, strike: float, time_to_expiry: float, risk_free_rate: float, option_type: str = 'call', min_vol: float = 0.0001, max_vol: float = 5.0) -> float:
    if market_price <= intrinsic_value(spot, strike, option_type):
        return min_vol
    low, high = min_vol, max_vol
    for _ in range(100):
        mid = (low + high) / 2.0
        price = black_scholes_price(spot, strike, time_to_expiry, risk_free_rate, mid, option_type)
        if price < market_price:
            low = mid
        else:
            high = mid
    return float((low + high) / 2.0)


def implied_volatility_percent(*args, **kwargs) -> float:
    return implied_volatility(*args, **kwargs) * 100.0


def volatility_spread(implied_vol: float, historical_vol: float) -> float:
    return float(implied_vol - historical_vol)


def volatility_spread_percent(implied_vol: float, historical_vol: float) -> float:
    return 100.0 * volatility_spread(implied_vol, historical_vol)


def classify_volatility_mispricing(implied_vol: float, historical_vol: float, threshold: float = 0.10) -> str:
    spread = implied_vol - historical_vol
    if spread > threshold:
        return 'expensive'
    if spread < -threshold:
        return 'cheap'
    return 'neutral'


def implied_volatility_summary(market_price: float, spot: float, strike: float, time_to_expiry: float, risk_free_rate: float, historical_vol: float, option_type: str = 'call', threshold: float = 0.10) -> dict:
    iv = implied_volatility(market_price, spot, strike, time_to_expiry, risk_free_rate, option_type)
    return {'implied_volatility': iv, 'volatility_spread': iv - historical_vol, 'classification': classify_volatility_mispricing(iv, historical_vol, threshold)}
