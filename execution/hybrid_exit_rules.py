"""
execution/hybrid_exit_rules.py

Hybrid exit rules for paper option positions.

This module decides when to close a paper option position using:
- hard stop loss
- near-expiry close
- hard take profit
- trailing profit protection
- trend/regime-based profit protection

Important:
This is for paper trading only.
It does not place real orders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class HybridExitConfig:
    """
    Hybrid exit configuration.

    stop_loss_pct:
        Hard stop loss. Example: -0.20 means close at -20%.

    soft_take_profit_pct:
        Profit level where model-based protection starts.
        Example: 0.35 means +35%.

    hard_take_profit_pct:
        Always close once profit reaches this level.
        Example: 0.60 means +60%.

    trailing_activation_pct:
        Start trailing once position reaches this profit.
        Example: 0.25 means activate trailing at +25%.

    trailing_drawdown_pct:
        Close if option falls this much from high watermark after trailing activates.
        Example: 0.15 means close after 15% drop from best seen option price.

    min_days_to_expiry:
        Close when DTE falls below this.
    """

    stop_loss_pct: float = -0.20
    soft_take_profit_pct: float = 0.35
    hard_take_profit_pct: float = 0.60
    trailing_activation_pct: float = 0.25
    trailing_drawdown_pct: float = 0.15
    min_days_to_expiry: float = 1.0

    close_profitable_trade_on_trend_loss: bool = True
    close_profitable_trade_on_hostile_regime: bool = True


@dataclass
class HybridExitDecision:
    should_close: bool
    reason: str
    pnl_pct: float
    highest_price_usd: float
    trailing_stop_price_usd: float
    trend_supportive: bool
    regime_hostile: bool


def safe_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default

    return bool(value)


def trend_supports_position(
    option_type: str,
    trend_regime: Any,
) -> bool:
    """
    Check whether current market trend still supports the position direction.

    Calls need bullish regime.
    Puts need bearish regime.
    """

    option_type = str(option_type).lower().strip()

    bullish = safe_bool(getattr(trend_regime, "bullish", False))
    bearish = safe_bool(getattr(trend_regime, "bearish", False))

    if option_type == "call":
        return bullish

    if option_type == "put":
        return bearish

    return False


def evaluate_hybrid_exit(
    option_type: str,
    entry_price_usd: float,
    current_price_usd: float,
    highest_price_usd: float,
    days_to_expiry: float,
    trend_regime: Any,
    config: HybridExitConfig | None = None,
) -> HybridExitDecision:
    """
    Decide whether to close a paper option position.

    Exit priority:
    1. Invalid data
    2. Near expiry
    3. Hard stop loss
    4. Hard take profit
    5. Trailing profit protection
    6. Profit protection when trend disappears
    7. Profit protection when regime becomes hostile
    """

    if config is None:
        config = HybridExitConfig()

    if entry_price_usd <= 0:
        return HybridExitDecision(
            should_close=True,
            reason="invalid_entry_price",
            pnl_pct=0.0,
            highest_price_usd=max(highest_price_usd, current_price_usd),
            trailing_stop_price_usd=0.0,
            trend_supportive=False,
            regime_hostile=False,
        )

    if current_price_usd <= 0:
        return HybridExitDecision(
            should_close=True,
            reason="invalid_current_price",
            pnl_pct=-1.0,
            highest_price_usd=max(highest_price_usd, current_price_usd),
            trailing_stop_price_usd=0.0,
            trend_supportive=False,
            regime_hostile=False,
        )

    updated_highest_price = max(
        float(highest_price_usd),
        float(current_price_usd),
        float(entry_price_usd),
    )

    pnl_pct = current_price_usd / entry_price_usd - 1.0

    trend_ok = trend_supports_position(
        option_type=option_type,
        trend_regime=trend_regime,
    )

    regime_hostile = safe_bool(getattr(trend_regime, "hostile", False))

    trailing_stop_price = updated_highest_price * (1.0 - config.trailing_drawdown_pct)

    # 1. Near expiry
    if days_to_expiry <= config.min_days_to_expiry:
        return HybridExitDecision(
            should_close=True,
            reason="near_expiry",
            pnl_pct=float(pnl_pct),
            highest_price_usd=float(updated_highest_price),
            trailing_stop_price_usd=float(trailing_stop_price),
            trend_supportive=trend_ok,
            regime_hostile=regime_hostile,
        )

    # 2. Hard stop loss
    if pnl_pct <= config.stop_loss_pct:
        return HybridExitDecision(
            should_close=True,
            reason="hard_stop_loss",
            pnl_pct=float(pnl_pct),
            highest_price_usd=float(updated_highest_price),
            trailing_stop_price_usd=float(trailing_stop_price),
            trend_supportive=trend_ok,
            regime_hostile=regime_hostile,
        )

    # 3. Hard take profit
    if pnl_pct >= config.hard_take_profit_pct:
        return HybridExitDecision(
            should_close=True,
            reason="hard_take_profit",
            pnl_pct=float(pnl_pct),
            highest_price_usd=float(updated_highest_price),
            trailing_stop_price_usd=float(trailing_stop_price),
            trend_supportive=trend_ok,
            regime_hostile=regime_hostile,
        )

    # 4. Trailing profit protection
    if pnl_pct >= config.trailing_activation_pct:
        if current_price_usd <= trailing_stop_price:
            return HybridExitDecision(
                should_close=True,
                reason="trailing_profit_stop",
                pnl_pct=float(pnl_pct),
                highest_price_usd=float(updated_highest_price),
                trailing_stop_price_usd=float(trailing_stop_price),
                trend_supportive=trend_ok,
                regime_hostile=regime_hostile,
            )

    # 5. Trend-loss profit protection
    if (
        config.close_profitable_trade_on_trend_loss
        and pnl_pct >= config.soft_take_profit_pct
        and not trend_ok
    ):
        return HybridExitDecision(
            should_close=True,
            reason="profit_protected_trend_lost",
            pnl_pct=float(pnl_pct),
            highest_price_usd=float(updated_highest_price),
            trailing_stop_price_usd=float(trailing_stop_price),
            trend_supportive=trend_ok,
            regime_hostile=regime_hostile,
        )

    # 6. Hostile-regime profit protection
    if (
        config.close_profitable_trade_on_hostile_regime
        and pnl_pct >= config.soft_take_profit_pct
        and regime_hostile
    ):
        return HybridExitDecision(
            should_close=True,
            reason="profit_protected_hostile_regime",
            pnl_pct=float(pnl_pct),
            highest_price_usd=float(updated_highest_price),
            trailing_stop_price_usd=float(trailing_stop_price),
            trend_supportive=trend_ok,
            regime_hostile=regime_hostile,
        )

    return HybridExitDecision(
        should_close=False,
        reason="hold",
        pnl_pct=float(pnl_pct),
        highest_price_usd=float(updated_highest_price),
        trailing_stop_price_usd=float(trailing_stop_price),
        trend_supportive=trend_ok,
        regime_hostile=regime_hostile,
    )