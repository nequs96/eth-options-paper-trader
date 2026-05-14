from __future__ import annotations
from dataclasses import dataclass


@dataclass
class HybridExitConfig:
    stop_loss_pct: float = -0.25
    emergency_stop_loss_pct: float = -0.45
    soft_take_profit_pct: float = 0.30
    hard_take_profit_pct: float = 0.80
    trailing_activation_pct: float = 0.25
    trailing_drawdown_pct: float = 0.18
    min_days_to_expiry: float = 1.5


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


def evaluate_hybrid_exit(option_type: str, entry_price_usd: float, current_price_usd: float, highest_price_usd: float, days_to_expiry: float, trend_regime=None, config: HybridExitConfig | None = None) -> HybridExitDecision:
    config = config or HybridExitConfig()
    entry = max(float(entry_price_usd), 1e-9)
    current = float(current_price_usd)
    high = max(float(highest_price_usd), entry, current)
    pnl = current / entry - 1.0
    trailing = high * (1.0 - config.trailing_drawdown_pct)
    close, reason = False, 'hold'
    if days_to_expiry <= config.min_days_to_expiry:
        close, reason = True, 'near_expiry'
    elif pnl <= config.emergency_stop_loss_pct:
        close, reason = True, 'emergency_stop_loss'
    elif pnl <= config.stop_loss_pct:
        close, reason = True, 'stop_loss'
    elif pnl >= config.hard_take_profit_pct:
        close, reason = True, 'hard_take_profit'
    elif high / entry - 1.0 >= config.trailing_activation_pct and current <= trailing:
        close, reason = True, 'trailing_profit_stop'
    return HybridExitDecision(close, reason, pnl, high, high / entry - 1.0, trailing, False, False, 0.0, False)
