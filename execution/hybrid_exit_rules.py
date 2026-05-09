"""
execution/hybrid_exit_rules.py

Hybrid exit rules for long option paper positions.

Pure decision module:
- no file I/O,
- no exchange calls,
- no cash/accounting logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class HybridExitConfig:
    stop_loss_pct: float = -0.25
    emergency_stop_loss_pct: float = -0.45
    soft_take_profit_pct: float = 0.30
    hard_take_profit_pct: float = 0.80
    trailing_activation_pct: float = 0.25
    trailing_drawdown_pct: float = 0.18
    min_days_to_expiry: float = 1.5

    trend_loss_profit_floor_pct: float = 0.12
    strong_trend_against_profit_floor_pct: float = 0.05
    volatility_contraction_profit_floor_pct: float = 0.18
    big_profit_giveback_activation_pct: float = 0.50
    big_profit_min_keep_pct: float = 0.22

    strong_trend_gap_threshold: float = 0.015
    volatility_contraction_ratio: float = 0.85

    close_profitable_trade_on_trend_loss: bool = True
    close_profitable_trade_on_hostile_regime: bool = True
    close_profitable_trade_on_volatility_contraction: bool = True


@dataclass
class HybridExitDecision:
    should_close: bool
    reason: str
    pnl_pct: float
    highest_price_usd: float
    highest_profit_pct: float
    trailing_stop_price_usd: float
    trend_supportive: bool
    regime_hostile: bool
    trend_strength: float
    volatility_contracting: bool


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:
        return default
    return float(number)


def safe_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    return bool(value)


def trend_strength_from_regime(trend_regime: Any) -> float:
    sma_fast = safe_float(getattr(trend_regime, "sma_fast", 0.0))
    sma_slow = safe_float(getattr(trend_regime, "sma_slow", 0.0))
    if sma_slow <= 0:
        return 0.0
    return float((sma_fast - sma_slow) / sma_slow)


def volatility_is_contracting(trend_regime: Any, config: HybridExitConfig) -> bool:
    hv_fast = safe_float(getattr(trend_regime, "hv_fast", 0.0))
    hv_slow = safe_float(getattr(trend_regime, "hv_slow", 0.0))
    if hv_fast <= 0 or hv_slow <= 0:
        return False
    return bool(hv_fast < hv_slow * config.volatility_contraction_ratio)


def trend_supports_position(option_type: str, trend_regime: Any) -> bool:
    option_type = str(option_type).lower().strip()
    bullish = safe_bool(getattr(trend_regime, "bullish", False))
    bearish = safe_bool(getattr(trend_regime, "bearish", False))

    if option_type == "call":
        return bullish
    if option_type == "put":
        return bearish
    return False


def strong_trend_against_position(
    option_type: str,
    trend_strength: float,
    config: HybridExitConfig,
) -> bool:
    option_type = str(option_type).lower().strip()
    if option_type == "call":
        return trend_strength <= -config.strong_trend_gap_threshold
    if option_type == "put":
        return trend_strength >= config.strong_trend_gap_threshold
    return False


def build_decision(
    should_close: bool,
    reason: str,
    pnl_pct: float,
    highest_price_usd: float,
    entry_price_usd: float,
    trailing_stop_price_usd: float,
    trend_supportive: bool,
    regime_hostile: bool,
    trend_strength: float,
    volatility_contracting: bool,
) -> HybridExitDecision:
    highest_profit_pct = highest_price_usd / entry_price_usd - 1.0 if entry_price_usd > 0 else 0.0
    return HybridExitDecision(
        should_close=bool(should_close),
        reason=str(reason),
        pnl_pct=float(pnl_pct),
        highest_price_usd=float(highest_price_usd),
        highest_profit_pct=float(highest_profit_pct),
        trailing_stop_price_usd=float(trailing_stop_price_usd),
        trend_supportive=bool(trend_supportive),
        regime_hostile=bool(regime_hostile),
        trend_strength=float(trend_strength),
        volatility_contracting=bool(volatility_contracting),
    )


def evaluate_hybrid_exit(
    option_type: str,
    entry_price_usd: float,
    current_price_usd: float,
    highest_price_usd: float,
    days_to_expiry: float,
    trend_regime: Any,
    config: HybridExitConfig | None = None,
) -> HybridExitDecision:
    if config is None:
        config = HybridExitConfig()

    entry = float(entry_price_usd)
    current = float(current_price_usd)
    previous_high = max(float(highest_price_usd), entry)
    updated_high = max(previous_high, current)

    if entry <= 0:
        return build_decision(True, "invalid_entry_price", 0.0, max(updated_high, 1.0), 1.0, 0.0, False, False, 0.0, False)
    if current <= 0:
        return build_decision(True, "invalid_current_price", -1.0, updated_high, entry, 0.0, False, False, 0.0, False)

    pnl_pct = current / entry - 1.0
    highest_profit_pct = updated_high / entry - 1.0
    trailing_stop = updated_high * (1.0 - config.trailing_drawdown_pct)

    trend_ok = trend_supports_position(option_type, trend_regime)
    hostile = safe_bool(getattr(trend_regime, "hostile", False))
    trend_strength = trend_strength_from_regime(trend_regime)
    vol_contracting = volatility_is_contracting(trend_regime, config)
    trend_against = strong_trend_against_position(option_type, trend_strength, config)

    if days_to_expiry <= config.min_days_to_expiry:
        return build_decision(True, "near_expiry", pnl_pct, updated_high, entry, trailing_stop, trend_ok, hostile, trend_strength, vol_contracting)
    if pnl_pct <= config.emergency_stop_loss_pct:
        return build_decision(True, "emergency_stop_loss", pnl_pct, updated_high, entry, trailing_stop, trend_ok, hostile, trend_strength, vol_contracting)
    if pnl_pct <= config.stop_loss_pct:
        return build_decision(True, "hard_stop_loss", pnl_pct, updated_high, entry, trailing_stop, trend_ok, hostile, trend_strength, vol_contracting)
    if pnl_pct >= config.hard_take_profit_pct:
        return build_decision(True, "hard_take_profit", pnl_pct, updated_high, entry, trailing_stop, trend_ok, hostile, trend_strength, vol_contracting)
    if highest_profit_pct >= config.big_profit_giveback_activation_pct and pnl_pct <= config.big_profit_min_keep_pct:
        return build_decision(True, "big_profit_giveback_protection", pnl_pct, updated_high, entry, trailing_stop, trend_ok, hostile, trend_strength, vol_contracting)
    if highest_profit_pct >= config.trailing_activation_pct and current <= trailing_stop:
        return build_decision(True, "trailing_profit_stop", pnl_pct, updated_high, entry, trailing_stop, trend_ok, hostile, trend_strength, vol_contracting)
    if config.close_profitable_trade_on_trend_loss and pnl_pct >= config.trend_loss_profit_floor_pct and not trend_ok:
        return build_decision(True, "profit_protected_trend_lost", pnl_pct, updated_high, entry, trailing_stop, trend_ok, hostile, trend_strength, vol_contracting)
    if pnl_pct >= config.strong_trend_against_profit_floor_pct and trend_against:
        return build_decision(True, "profit_protected_strong_trend_against", pnl_pct, updated_high, entry, trailing_stop, trend_ok, hostile, trend_strength, vol_contracting)
    if config.close_profitable_trade_on_hostile_regime and pnl_pct >= config.soft_take_profit_pct and hostile:
        return build_decision(True, "profit_protected_hostile_regime", pnl_pct, updated_high, entry, trailing_stop, trend_ok, hostile, trend_strength, vol_contracting)
    if config.close_profitable_trade_on_volatility_contraction and pnl_pct >= config.volatility_contraction_profit_floor_pct and vol_contracting:
        return build_decision(True, "profit_protected_volatility_contracting", pnl_pct, updated_high, entry, trailing_stop, trend_ok, hostile, trend_strength, vol_contracting)

    return build_decision(False, "hold", pnl_pct, updated_high, entry, trailing_stop, trend_ok, hostile, trend_strength, vol_contracting)
