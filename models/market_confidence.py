"""Market Confidence Index for ETH option entries.

The Market Confidence Index (MCI) combines edge quality, ETH regime, volatility
expansion, liquidity, Greeks, and portfolio safety into one dynamic score.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

import pandas as pd

from models.eth_forward_volatility import VolatilityForecast


@dataclass(frozen=True)
class MarketConfidence:
    mci: float
    edge_score: float
    regime_score: float
    vol_score: float
    liquidity_score: float
    greek_score: float
    portfolio_score: float
    required_price_edge: float
    required_vol_edge: float
    expected_return_hurdle: float
    reject_reason: str = ""


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def score_edge(price_diff_pct: float, vol_edge: float, spread_pct: float) -> float:
    # Negative price_diff_pct means market price below model price in the existing pipeline.
    cheap_edge = max(-price_diff_pct, 0.0)
    raw = 0.55 * clamp(cheap_edge / 0.25, 0.0, 1.0) + 0.45 * clamp(vol_edge / 0.25, 0.0, 1.0)
    penalty = clamp(spread_pct / 0.30, 0.0, 1.0) * 0.35
    return clamp(raw - penalty, 0.0, 1.0)


def score_regime(option_type: str, trend_regime: Any) -> float:
    option_type = str(option_type).lower().strip()
    bullish = bool(getattr(trend_regime, "bullish", False))
    bearish = bool(getattr(trend_regime, "bearish", False))
    hostile = bool(getattr(trend_regime, "hostile", False))
    sma_fast = safe_float(getattr(trend_regime, "sma_fast", 0.0))
    sma_slow = safe_float(getattr(trend_regime, "sma_slow", 0.0))
    trend_strength = (sma_fast - sma_slow) / sma_slow if sma_slow > 0 else 0.0

    directional = 0.5
    if option_type == "call":
        directional = 0.75 if bullish else 0.25 if bearish else 0.50
        directional += clamp(trend_strength / 0.08, -0.25, 0.25)
    elif option_type == "put":
        directional = 0.75 if bearish else 0.25 if bullish else 0.50
        directional += clamp(-trend_strength / 0.08, -0.25, 0.25)

    if hostile:
        directional *= 0.55
    return clamp(directional, 0.0, 1.0)


def score_liquidity(market_price_usd: float, spread_pct: float, open_interest: float | None = None) -> float:
    price_score = clamp((market_price_usd - 5.0) / 45.0, 0.0, 1.0)
    spread_score = clamp(1.0 - spread_pct / 0.30, 0.0, 1.0)
    oi_score = 0.5 if open_interest is None else clamp(math.log1p(max(open_interest, 0.0)) / math.log1p(500.0), 0.0, 1.0)
    return clamp(0.25 * price_score + 0.55 * spread_score + 0.20 * oi_score, 0.0, 1.0)


def score_greeks(delta: float | None, theta: float | None, vega: float | None, days_to_expiry: float, vol_expansion_score: float) -> float:
    dte_score = clamp((days_to_expiry - 3.0) / 25.0, 0.0, 1.0)

    if delta is None:
        delta_score = 0.55
    else:
        abs_delta = abs(delta)
        if 0.25 <= abs_delta <= 0.60:
            delta_score = 1.0
        elif 0.15 <= abs_delta < 0.25 or 0.60 < abs_delta <= 0.75:
            delta_score = 0.70
        elif 0.08 <= abs_delta < 0.15:
            delta_score = 0.40 + 0.30 * vol_expansion_score
        else:
            delta_score = 0.25

    theta_score = 0.65 if theta is None else clamp(1.0 - abs(theta) / 80.0, 0.20, 1.0)
    vega_score = 0.55 if vega is None else clamp(abs(vega) / 40.0, 0.25, 1.0)
    vega_score = 0.50 * vega_score + 0.50 * vol_expansion_score
    return clamp(0.35 * delta_score + 0.25 * theta_score + 0.20 * vega_score + 0.20 * dte_score, 0.0, 1.0)


def score_portfolio(current_drawdown: float, open_risk_pct: float, same_direction_count: int = 0, same_expiry_count: int = 0) -> float:
    dd_penalty = clamp(abs(min(current_drawdown, 0.0)) / 0.08, 0.0, 1.0)
    risk_penalty = clamp(open_risk_pct / 0.10, 0.0, 1.0)
    concentration_penalty = clamp(0.12 * same_direction_count + 0.10 * same_expiry_count, 0.0, 0.60)
    return clamp(1.0 - 0.45 * dd_penalty - 0.35 * risk_penalty - concentration_penalty, 0.0, 1.0)


def dynamic_required_edges(spread_pct: float, vol_uncertainty: float, regime_score: float, theta_pressure: float = 0.0) -> tuple[float, float, float]:
    liquidity_penalty = clamp(spread_pct / 0.30, 0.0, 1.0)
    regime_uncertainty = 1.0 - clamp(regime_score, 0.0, 1.0)
    required_price_edge = 0.06 + 0.10 * liquidity_penalty + 0.08 * clamp(theta_pressure, 0.0, 1.0) + 0.06 * regime_uncertainty
    required_vol_edge = 0.04 + 0.08 * clamp(vol_uncertainty, 0.0, 1.0) + 0.05 * liquidity_penalty
    expected_return_hurdle = 0.08 + 0.12 * liquidity_penalty + 0.10 * clamp(vol_uncertainty, 0.0, 1.0)
    return required_price_edge, required_vol_edge, expected_return_hurdle


def calculate_market_confidence(
    row: pd.Series,
    vol_forecast: VolatilityForecast,
    trend_regime: Any,
    current_drawdown: float = 0.0,
    open_risk_pct: float = 0.0,
    same_direction_count: int = 0,
    same_expiry_count: int = 0,
) -> MarketConfidence:
    option_type = str(row.get("option_type", "")).lower().strip()
    market_price = safe_float(row.get("market_price_usd", row.get("entry_price_usd", 0.0)))
    implied_vol = safe_float(row.get("implied_volatility", row.get("mark_iv", row.get("iv", 0.0))))
    price_diff_pct = safe_float(row.get("price_diff_pct", row.get("price_difference_percent", 0.0)))
    spread_pct = safe_float(row.get("bid_ask_spread_pct", 0.0))
    days_to_expiry = safe_float(row.get("days_to_expiry", 0.0))

    delta_raw = row.get("delta", None)
    theta_raw = row.get("theta", None)
    vega_raw = row.get("vega", None)
    oi_raw = row.get("open_interest", None)
    delta = None if delta_raw is None or pd.isna(delta_raw) else safe_float(delta_raw)
    theta = None if theta_raw is None or pd.isna(theta_raw) else safe_float(theta_raw)
    vega = None if vega_raw is None or pd.isna(vega_raw) else safe_float(vega_raw)
    open_interest = None if oi_raw is None or pd.isna(oi_raw) else safe_float(oi_raw)

    vol_edge = max(vol_forecast.forecast_vol - implied_vol, -safe_float(row.get("volatility_spread", 0.0)))
    vol_uncertainty = clamp(abs(vol_forecast.vol_zscore) / 3.0 + vol_forecast.vol_of_vol / 1.0, 0.0, 1.0)

    regime = score_regime(option_type, trend_regime)
    required_price_edge, required_vol_edge, hurdle = dynamic_required_edges(spread_pct, vol_uncertainty, regime)

    edge = score_edge(price_diff_pct, vol_edge, spread_pct)
    vol_score = clamp(0.65 * vol_forecast.expansion_score + 0.35 * clamp(vol_edge / max(required_vol_edge, 1e-6), 0.0, 1.0), 0.0, 1.0)
    liquidity = score_liquidity(market_price, spread_pct, open_interest)
    greek = score_greeks(delta, theta, vega, days_to_expiry, vol_score)
    portfolio = score_portfolio(current_drawdown, open_risk_pct, same_direction_count, same_expiry_count)

    mci = 0.25 * edge + 0.20 * regime + 0.20 * vol_score + 0.15 * liquidity + 0.10 * greek + 0.10 * portfolio
    mci = clamp(mci, 0.0, 1.0)

    reject_reason = ""
    if market_price <= 0:
        reject_reason = "invalid_market_price"
    elif days_to_expiry < 3.0:
        reject_reason = "too_close_to_expiry"
    elif liquidity < 0.25:
        reject_reason = "liquidity_too_weak"
    elif edge < 0.25:
        reject_reason = "edge_too_weak_after_costs"
    elif mci < 0.35:
        reject_reason = "market_confidence_too_low"

    return MarketConfidence(
        mci=float(mci),
        edge_score=float(edge),
        regime_score=float(regime),
        vol_score=float(vol_score),
        liquidity_score=float(liquidity),
        greek_score=float(greek),
        portfolio_score=float(portfolio),
        required_price_edge=float(required_price_edge),
        required_vol_edge=float(required_vol_edge),
        expected_return_hurdle=float(hurdle),
        reject_reason=reject_reason,
    )
