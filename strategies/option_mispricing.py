from __future__ import annotations
from dataclasses import dataclass
from models.black_scholes import black_scholes_price
from models.implied_volatility import implied_volatility


@dataclass
class MispricingResult:
    option_type: str
    spot_price: float
    strike_price: float
    time_to_expiry: float
    risk_free_rate: float
    market_price: float
    theoretical_price: float
    price_difference: float
    price_difference_percent: float
    implied_volatility: float
    historical_volatility: float
    volatility_spread: float
    volatility_spread_percent_points: float
    classification: str
    signal: str
    confidence_level: str
    mispricing_score: float
    explanation: str


def classify_option_mispricing(price_difference_percent: float | None = None, vol_spread: float | None = None, price_threshold: float = 0.05, volatility_threshold: float = 0.05, price_diff_pct: float | None = None, vol_threshold: float | None = None) -> str:
    price_edge = price_diff_pct if price_diff_pct is not None else (price_difference_percent or 0.0)
    vol_edge = vol_spread or 0.0
    vol_limit = vol_threshold if vol_threshold is not None else volatility_threshold
    if price_edge <= -price_threshold and vol_edge <= -vol_limit:
        return 'cheap'
    if price_edge >= price_threshold and vol_edge >= vol_limit:
        return 'expensive'
    return 'neutral'


def analyze_option_mispricing(market_price: float, spot_price: float, strike_price: float, time_to_expiry: float, risk_free_rate: float, historical_volatility: float, option_type: str = 'call', price_threshold: float = 0.05, volatility_threshold: float = 0.05) -> MispricingResult:
    theoretical = black_scholes_price(spot_price, strike_price, time_to_expiry, risk_free_rate, historical_volatility, option_type)
    price_diff = market_price - theoretical
    price_diff_pct = price_diff / theoretical if theoretical > 0 else 0.0
    iv = implied_volatility(market_price, spot_price, strike_price, time_to_expiry, risk_free_rate, option_type)
    spread = iv - historical_volatility
    classification = classify_option_mispricing(price_diff_pct, spread, price_threshold, volatility_threshold)
    score = abs(price_diff_pct) + abs(spread)
    return MispricingResult(option_type, spot_price, strike_price, time_to_expiry, risk_free_rate, market_price, theoretical, price_diff, price_diff_pct, iv, historical_volatility, spread, spread * 100.0, classification, classification, 'medium', score, 'ok')
