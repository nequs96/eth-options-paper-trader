from __future__ import annotations
from dataclasses import dataclass
import math
import pandas as pd


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
    reject_reason: str = ''


def safe_float(value, default: float = 0.0) -> float:
    try:
        result = float(value)
    except Exception:
        return default
    return result if math.isfinite(result) else default


def clean_string(value) -> str:
    if value is None:
        return ''
    try:
        if pd.isna(value):
            return ''
    except Exception:
        pass
    text = str(value).strip()
    return '' if text.lower() in {'nan', 'none', 'null'} else text


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def calculate_market_confidence(row, vol_forecast=None, trend_regime=None, current_drawdown: float = 0.0, open_risk_pct: float = 0.0, same_direction_count: int = 0, same_expiry_count: int = 0) -> MarketConfidence:
    price = safe_float(row.get('market_price_usd'))
    price_diff = safe_float(row.get('price_diff_pct'))
    spread = safe_float(row.get('bid_ask_spread_pct'))
    dte = safe_float(row.get('days_to_expiry'))
    iv = safe_float(row.get('implied_volatility', row.get('mark_iv', 0.0)))
    forecast_vol = safe_float(getattr(vol_forecast, 'forecast_vol', 0.0))
    expansion = safe_float(getattr(vol_forecast, 'expansion_score', 0.5), 0.5)
    edge = clamp(max(-price_diff, 0.0) / 0.35 - spread / 0.50, 0.0, 1.0)
    vol = clamp(expansion * 0.60 + max(forecast_vol - iv, 0.0) / 0.50 * 0.40, 0.0, 1.0)
    liquidity = clamp(1.0 - spread / 0.35, 0.0, 1.0)
    greek = clamp((dte - 2.0) / 30.0, 0.0, 1.0)
    portfolio = clamp(1.0 - open_risk_pct / 0.10, 0.0, 1.0)
    regime = 0.55
    mci = clamp(0.25 * edge + 0.20 * vol + 0.15 * regime + 0.15 * liquidity + 0.10 * greek + 0.15 * portfolio, 0.0, 1.0)
    reason = ''
    if price <= 0:
        reason = 'invalid_market_price'
    elif dte < 3:
        reason = 'too_close_to_expiry'
    elif edge < 0.20:
        reason = 'edge_too_weak_after_costs'
    elif mci < 0.35:
        reason = 'market_confidence_too_low'
    return MarketConfidence(mci, edge, regime, vol, liquidity, greek, portfolio, 0.06, 0.04, 0.08, reason)
