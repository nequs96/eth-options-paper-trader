from __future__ import annotations
import math

SQRT2 = math.sqrt(2.0)
SQRT_2PI = math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(float(x) / SQRT2))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * float(x) * float(x)) / SQRT_2PI


def intrinsic_value(spot: float, strike: float, option_type: str) -> float:
    option_type = str(option_type).lower().strip()
    return float(max(spot - strike, 0.0) if option_type in {'call', 'c'} else max(strike - spot, 0.0))


def calculate_d1(spot: float, strike: float, time_to_expiry: float, risk_free_rate: float, volatility: float) -> float:
    if spot <= 0 or strike <= 0 or time_to_expiry <= 0 or volatility <= 0:
        raise ValueError('Black-Scholes inputs must be positive for spot, strike, time, and volatility.')
    return (math.log(spot / strike) + (risk_free_rate + 0.5 * volatility * volatility) * time_to_expiry) / (volatility * math.sqrt(time_to_expiry))


def calculate_d2(spot: float, strike: float, time_to_expiry: float, risk_free_rate: float, volatility: float) -> float:
    return calculate_d1(spot, strike, time_to_expiry, risk_free_rate, volatility) - volatility * math.sqrt(time_to_expiry)


def black_scholes_call(spot: float, strike: float, time_to_expiry: float, risk_free_rate: float, volatility: float) -> float:
    d1 = calculate_d1(spot, strike, time_to_expiry, risk_free_rate, volatility)
    d2 = d1 - volatility * math.sqrt(time_to_expiry)
    return float(spot * _norm_cdf(d1) - strike * math.exp(-risk_free_rate * time_to_expiry) * _norm_cdf(d2))


def black_scholes_put(spot: float, strike: float, time_to_expiry: float, risk_free_rate: float, volatility: float) -> float:
    d1 = calculate_d1(spot, strike, time_to_expiry, risk_free_rate, volatility)
    d2 = d1 - volatility * math.sqrt(time_to_expiry)
    return float(strike * math.exp(-risk_free_rate * time_to_expiry) * _norm_cdf(-d2) - spot * _norm_cdf(-d1))


def black_scholes_price(spot: float, strike: float, time_to_expiry: float, risk_free_rate: float, volatility: float, option_type: str = 'call') -> float:
    if time_to_expiry <= 0 or volatility <= 0:
        return intrinsic_value(spot, strike, option_type)
    option_type = str(option_type).lower().strip()
    price = black_scholes_call(spot, strike, time_to_expiry, risk_free_rate, volatility) if option_type in {'call', 'c'} else black_scholes_put(spot, strike, time_to_expiry, risk_free_rate, volatility)
    return float(max(price, 0.0))


def time_value(spot: float, strike: float, time_to_expiry: float, risk_free_rate: float, volatility: float, option_type: str = 'call') -> float:
    return max(black_scholes_price(spot, strike, time_to_expiry, risk_free_rate, volatility, option_type) - intrinsic_value(spot, strike, option_type), 0.0)
