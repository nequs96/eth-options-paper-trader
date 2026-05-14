from __future__ import annotations
import math
from models.black_scholes import calculate_d1, calculate_d2, _norm_cdf, _norm_pdf


def delta(spot: float, strike: float, time_to_expiry: float, risk_free_rate: float, volatility: float, option_type: str = 'call') -> float:
    d1 = calculate_d1(spot, strike, time_to_expiry, risk_free_rate, volatility)
    return float(_norm_cdf(d1) if str(option_type).lower() in {'call', 'c'} else _norm_cdf(d1) - 1.0)


def gamma(spot: float, strike: float, time_to_expiry: float, risk_free_rate: float, volatility: float) -> float:
    d1 = calculate_d1(spot, strike, time_to_expiry, risk_free_rate, volatility)
    return float(_norm_pdf(d1) / (spot * volatility * math.sqrt(time_to_expiry)))


def vega(spot: float, strike: float, time_to_expiry: float, risk_free_rate: float, volatility: float) -> float:
    d1 = calculate_d1(spot, strike, time_to_expiry, risk_free_rate, volatility)
    return float(spot * _norm_pdf(d1) * math.sqrt(time_to_expiry) / 100.0)


def theta(spot: float, strike: float, time_to_expiry: float, risk_free_rate: float, volatility: float, option_type: str = 'call') -> float:
    d1 = calculate_d1(spot, strike, time_to_expiry, risk_free_rate, volatility)
    d2 = calculate_d2(spot, strike, time_to_expiry, risk_free_rate, volatility)
    first = -(spot * _norm_pdf(d1) * volatility) / (2.0 * math.sqrt(time_to_expiry))
    option_type = str(option_type).lower()
    if option_type in {'call', 'c'}:
        second = -risk_free_rate * strike * math.exp(-risk_free_rate * time_to_expiry) * _norm_cdf(d2)
    else:
        second = risk_free_rate * strike * math.exp(-risk_free_rate * time_to_expiry) * _norm_cdf(-d2)
    return float((first + second) / 365.0)


def rho(spot: float, strike: float, time_to_expiry: float, risk_free_rate: float, volatility: float, option_type: str = 'call') -> float:
    d2 = calculate_d2(spot, strike, time_to_expiry, risk_free_rate, volatility)
    option_type = str(option_type).lower()
    if option_type in {'call', 'c'}:
        return float(strike * time_to_expiry * math.exp(-risk_free_rate * time_to_expiry) * _norm_cdf(d2) / 100.0)
    return float(-strike * time_to_expiry * math.exp(-risk_free_rate * time_to_expiry) * _norm_cdf(-d2) / 100.0)


def all_greeks(spot: float, strike: float, time_to_expiry: float, risk_free_rate: float, volatility: float, option_type: str = 'call') -> dict[str, float]:
    return {
        'delta': delta(spot, strike, time_to_expiry, risk_free_rate, volatility, option_type),
        'gamma': gamma(spot, strike, time_to_expiry, risk_free_rate, volatility),
        'vega': vega(spot, strike, time_to_expiry, risk_free_rate, volatility),
        'theta': theta(spot, strike, time_to_expiry, risk_free_rate, volatility, option_type),
        'rho': rho(spot, strike, time_to_expiry, risk_free_rate, volatility, option_type),
    }
