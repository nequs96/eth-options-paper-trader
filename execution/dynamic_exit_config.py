from __future__ import annotations
from dataclasses import dataclass
from models.market_confidence import safe_float
from execution.hybrid_exit_rules import HybridExitConfig


@dataclass(frozen=True)
class DynamicExitPlan:
    stop_loss_pct: float
    emergency_stop_loss_pct: float
    soft_take_profit_pct: float
    hard_take_profit_pct: float
    trailing_activation_pct: float
    trailing_drawdown_pct: float
    trend_loss_profit_floor_pct: float = 0.12
    strong_trend_against_profit_floor_pct: float = 0.05
    volatility_contraction_profit_floor_pct: float = 0.18
    big_profit_giveback_activation_pct: float = 0.50
    big_profit_min_keep_pct: float = 0.22
    min_days_to_expiry: float = 1.5

    def to_hybrid_config(self) -> HybridExitConfig:
        return HybridExitConfig(self.stop_loss_pct, self.emergency_stop_loss_pct, self.soft_take_profit_pct, self.hard_take_profit_pct, self.trailing_activation_pct, self.trailing_drawdown_pct, self.min_days_to_expiry)


def build_dynamic_exit_plan(confidence, days_to_expiry: float, theta_pressure: float = 0.0) -> DynamicExitPlan:
    return DynamicExitPlan(-0.25, -0.45, 0.30, 0.80, 0.25, 0.18, min_days_to_expiry=1.0 if days_to_expiry < 7 else 1.5)


def exit_plan_columns(plan: DynamicExitPlan) -> dict:
    return {'dynamic_stop_loss_pct': plan.stop_loss_pct, 'dynamic_emergency_stop_loss_pct': plan.emergency_stop_loss_pct, 'dynamic_soft_take_profit_pct': plan.soft_take_profit_pct, 'dynamic_hard_take_profit_pct': plan.hard_take_profit_pct, 'dynamic_trailing_activation_pct': plan.trailing_activation_pct, 'dynamic_trailing_drawdown_pct': plan.trailing_drawdown_pct, 'dynamic_min_days_to_expiry_exit': plan.min_days_to_expiry}


def exit_plan_from_position(position) -> HybridExitConfig:
    get = position.get if hasattr(position, 'get') else lambda key, default=None: default
    return HybridExitConfig(safe_float(get('dynamic_stop_loss_pct', -0.25), -0.25), safe_float(get('dynamic_emergency_stop_loss_pct', -0.45), -0.45), safe_float(get('dynamic_soft_take_profit_pct', 0.30), 0.30), safe_float(get('dynamic_hard_take_profit_pct', 0.80), 0.80), safe_float(get('dynamic_trailing_activation_pct', 0.25), 0.25), safe_float(get('dynamic_trailing_drawdown_pct', 0.18), 0.18), safe_float(get('dynamic_min_days_to_expiry_exit', 1.5), 1.5))
