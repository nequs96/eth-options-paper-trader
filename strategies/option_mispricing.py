"""
strategies/option_mispricing.py

Robust ETH options mispricing research signal.

This module compares:
- market option price
- Black-Scholes theoretical price
- implied volatility
- historical volatility

The goal is to classify an option as:
- expensive
- cheap
- neutral

This is a research signal only.
NO trades are placed here.
"""

from dataclasses import dataclass
import math

from models.black_scholes import black_scholes_price
from models.implied_volatility import (
    implied_volatility,
    volatility_spread,
    volatility_spread_percent,
)

# =========================
# Configuration
# =========================

MIN_OPTION_PRICE = 0.0001
MIN_TIME_TO_EXPIRY = 1 / 365        # 1 day
MAX_TIME_TO_EXPIRY = 2.0            # 2 years

MIN_VOLATILITY = 0.05
MAX_VOLATILITY = 3.00

MIN_PRICE_THRESHOLD = 0.05          # 5%
MIN_VOL_THRESHOLD = 0.05            # 5 vol points

HIGH_CONFIDENCE_PRICE = 0.20
HIGH_CONFIDENCE_VOL = 0.20


# =========================
# Result container
# =========================

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


# =========================
# Utilities
# =========================

def _is_valid_number(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def _safe_reject(reason: str) -> MispricingResult:
    return MispricingResult(
        option_type="unknown",
        spot_price=0.0,
        strike_price=0.0,
        time_to_expiry=0.0,
        risk_free_rate=0.0,
        market_price=0.0,
        theoretical_price=0.0,
        price_difference=0.0,
        price_difference_percent=0.0,
        implied_volatility=0.0,
        historical_volatility=0.0,
        volatility_spread=0.0,
        volatility_spread_percent_points=0.0,
        classification="invalid",
        signal="invalid_data",
        confidence_level="none",
        mispricing_score=0.0,
        explanation=reason,
    )


# =========================
# Core calculations
# =========================

def calculate_price_difference(market_price: float, theoretical_price: float) -> float:
    return float(market_price - theoretical_price)


def calculate_price_difference_percent(market_price: float, theoretical_price: float) -> float:
    if theoretical_price <= 0:
        return 0.0
    return float((market_price - theoretical_price) / theoretical_price)


# ✅ BACKWARD‑COMPATIBLE CLASSIFIER
def classify_option_mispricing(
    price_difference_percent: float | None = None,
    vol_spread: float | None = None,
    price_threshold: float = MIN_PRICE_THRESHOLD,
    volatility_threshold: float = MIN_VOL_THRESHOLD,
    price_diff_pct: float | None = None,
    vol_threshold: float | None = None,
) -> str:
    """
    Classify option as expensive, cheap, or neutral.

    Supports ALL call styles used across the project.
    """

    if price_difference_percent is None:
        price_difference_percent = price_diff_pct

    if vol_threshold is not None:
        volatility_threshold = vol_threshold

    if price_difference_percent is None or vol_spread is None:
        raise ValueError("price_difference_percent and vol_spread are required")

    expensive_by_price = price_difference_percent > price_threshold
    expensive_by_vol = vol_spread > volatility_threshold

    cheap_by_price = price_difference_percent < -price_threshold
    cheap_by_vol = vol_spread < -volatility_threshold

    if expensive_by_price and expensive_by_vol:
        return "expensive"

    if cheap_by_price and cheap_by_vol:
        return "cheap"

    return "neutral"


def confidence_from_magnitude(price_diff_pct: float, vol_spread: float) -> str:
    if abs(price_diff_pct) >= HIGH_CONFIDENCE_PRICE and abs(vol_spread) >= HIGH_CONFIDENCE_VOL:
        return "high"
    if abs(price_diff_pct) >= MIN_PRICE_THRESHOLD and abs(vol_spread) >= MIN_VOL_THRESHOLD:
        return "medium"
    return "low"


def compute_mispricing_score(price_diff_pct: float, vol_spread: float) -> float:
    score = 0.5 * price_diff_pct + 0.5 * vol_spread
    return max(-1.0, min(1.0, float(score)))


# =========================
# Main analysis
# =========================

def analyze_option_mispricing(
    market_price: float,
    spot_price: float,
    strike_price: float,
    time_to_expiry: float,
    risk_free_rate: float,
    historical_volatility: float,
    option_type: str = "call",
    price_threshold: float = MIN_PRICE_THRESHOLD,
    volatility_threshold: float = MIN_VOL_THRESHOLD,
) -> MispricingResult:

    values = [
        market_price, spot_price, strike_price,
        time_to_expiry, historical_volatility
    ]

    if not all(_is_valid_number(v) for v in values):
        return _safe_reject("Invalid numeric input")

    if market_price < MIN_OPTION_PRICE:
        return _safe_reject("Market price too small")

    if spot_price <= 0 or strike_price <= 0:
        return _safe_reject("Invalid spot or strike price")

    if not (MIN_TIME_TO_EXPIRY <= time_to_expiry <= MAX_TIME_TO_EXPIRY):
        return _safe_reject("Time to expiry out of range")

    if not (MIN_VOLATILITY <= historical_volatility <= MAX_VOLATILITY):
        return _safe_reject("Historical volatility out of bounds")

    option_type = option_type.lower().strip()
    if option_type not in {"call", "put"}:
        return _safe_reject("Invalid option type")

    theoretical_price = black_scholes_price(
        S=spot_price,
        K=strike_price,
        T=time_to_expiry,
        r=risk_free_rate,
        sigma=historical_volatility,
        option_type=option_type,
    )

    if not _is_valid_number(theoretical_price) or theoretical_price <= 0:
        return _safe_reject("Invalid theoretical price")

    iv = implied_volatility(
        market_price=market_price,
        S=spot_price,
        K=strike_price,
        T=time_to_expiry,
        r=risk_free_rate,
        option_type=option_type,
    )

    if not _is_valid_number(iv) or not (MIN_VOLATILITY <= iv <= MAX_VOLATILITY):
        return _safe_reject("Invalid implied volatility")

    price_diff = calculate_price_difference(market_price, theoretical_price)
    price_diff_pct = calculate_price_difference_percent(market_price, theoretical_price)

    vol_sp = volatility_spread(iv, historical_volatility)
    vol_sp_pp = volatility_spread_percent(iv, historical_volatility)

    classification = classify_option_mispricing(
        price_difference_percent=price_diff_pct,
        vol_spread=vol_sp,
        price_threshold=price_threshold,
        volatility_threshold=volatility_threshold,
    )

    confidence = confidence_from_magnitude(price_diff_pct, vol_sp)
    score = compute_mispricing_score(price_diff_pct, vol_sp)

    signal = (
        f"{option_type}_looks_{classification}"
        if classification in {"cheap", "expensive"}
        else "no_clear_edge"
    )

    explanation = (
        f"Classification={classification}, "
        f"price_diff={price_diff_pct:.2%}, "
        f"vol_spread={vol_sp_pp:.2f}pp, "
        f"confidence={confidence}"
    )

    return MispricingResult(
        option_type=option_type,
        spot_price=float(spot_price),
        strike_price=float(strike_price),
        time_to_expiry=float(time_to_expiry),
        risk_free_rate=float(risk_free_rate),
        market_price=float(market_price),
        theoretical_price=float(theoretical_price),
        price_difference=float(price_diff),
        price_difference_percent=float(price_diff_pct),
        implied_volatility=float(iv),
        historical_volatility=float(historical_volatility),
        volatility_spread=float(vol_sp),
        volatility_spread_percent_points=float(vol_sp_pp),
        classification=classification,
        signal=signal,
        confidence_level=confidence,
        mispricing_score=score,
        explanation=explanation,
    )