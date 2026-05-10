"""Dynamic exit configuration for ETH option paper positions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from models.market_confidence import MarketConfidence, clamp, safe_float
from execution.hybrid_exit_rules import HybridExitConfig


@dataclass(frozen=True)
class DynamicExitPlan:
    stop_loss_pct: float
    emergency_stop_loss_pct: float
    soft_take_profit_pct: float
    hard_take_profit_pct: float
    trailing_activation_pct: float
    trailing_drawdown_pct: float
    trend_loss_profit_floor_pct: float
    strong_trend_against_profit_floor_pct: float
    volatility_contraction_profit_floor_pct: float
    big_profit_giveback_activation_pct: float
    big_profit_min_keep_pct: float
    min_days_to_expiry: float

    def to_hybrid_config(self) -> HybridExitConfig:
        return HybridExitConfig(
            stop_loss_pct=self.stop_loss_pct,
            emergency_stop_loss_pct=self.emergency_stop_loss_pct,
            soft_take_profit_pct=self.soft_take_profit_pct,
            hard_take_profit_pct=self.hard_take_profit_pct,
            trailing_activation_pct=self.trailing_activation_pct,
            trailing_drawdown_pct=self.trailing_drawdown_pct,
            trend_loss_profit_floor_pct=self.trend_loss_profit_floor_pct,
            strong_trend_against_profit_floor_pct=self.strong_trend_against_profit_floor_pct,
            volatility_contraction_profit_floor_pct=self.volatility_contraction_profit_floor_pct,
            big_profit_giveback_activation_pct=self.big_profit_giveback_activation_pct,
            big_profit_min_keep_pct=self.big_profit_min_keep_pct,
            min_days_to_expiry=self.min_days_to_expiry,
        )


def build_dynamic_exit_plan(confidence: MarketConfidence, days_to_expiry: float, theta_pressure: float = 0.0) -> DynamicExitPlan:
    mci = confidence.mci
    vol = confidence.vol_score
    trend = confidence.regime_score
    liquidity_penalty = 1.0 - confidence.liquidity_score
    theta_pressure = clamp(theta_pressure, 0.0, 1.0)

    stop_loss = clamp(-0.18 - 0.14 * (1.0 - mci) - 0.08 * liquidity_penalty - 0.06 * theta_pressure, -0.50, -0.18)
    emergency = clamp(stop_loss - 0.18, -0.70, -0.35)
    soft_tp = clamp(0.18 + 0.35 * mci + 0.20 * vol + 0.15 * trend, 0.25, 0.90)
    hard_tp = clamp(0.45 + 0.90 * mci + 0.45 * vol + 0.25 * trend, 0.60, 2.20)
    trailing_activation = clamp(0.15 + 0.30 * mci, 0.20, 0.50)
    trailing_drawdown = clamp(0.12 + 0.12 * mci + 0.08 * vol, 0.12, 0.32)
    trend_loss_floor = clamp(0.08 + 0.15 * mci, 0.08, 0.30)
    strong_against_floor = clamp(0.03 + 0.10 * mci, 0.03, 0.18)
    vol_contract_floor = clamp(0.10 + 0.18 * mci, 0.10, 0.35)
    giveback_activation = clamp(0.35 + 0.35 * mci + 0.20 * vol, 0.35, 0.90)
    min_keep = clamp(0.12 + 0.25 * mci, 0.12, 0.40)
    min_dte = 1.0 if days_to_expiry <= 7 else 1.5 if days_to_expiry <= 21 else 2.0

    return DynamicExitPlan(stop_loss, emergency, soft_tp, hard_tp, trailing_activation, trailing_drawdown, trend_loss_floor, strong_against_floor, vol_contract_floor, giveback_activation, min_keep, min_dte)


def exit_plan_columns(plan: DynamicExitPlan) -> dict[str, float]:
    return {
        "dynamic_stop_loss_pct": plan.stop_loss_pct,
        "dynamic_emergency_stop_loss_pct": plan.emergency_stop_loss_pct,
        "dynamic_soft_take_profit_pct": plan.soft_take_profit_pct,
        "dynamic_hard_take_profit_pct": plan.hard_take_profit_pct,
        "dynamic_trailing_activation_pct": plan.trailing_activation_pct,
        "dynamic_trailing_drawdown_pct": plan.trailing_drawdown_pct,
        "dynamic_trend_loss_profit_floor_pct": plan.trend_loss_profit_floor_pct,
        "dynamic_strong_trend_against_profit_floor_pct": plan.strong_trend_against_profit_floor_pct,
        "dynamic_volatility_contraction_profit_floor_pct": plan.volatility_contraction_profit_floor_pct,
        "dynamic_big_profit_giveback_activation_pct": plan.big_profit_giveback_activation_pct,
        "dynamic_big_profit_min_keep_pct": plan.big_profit_min_keep_pct,
        "dynamic_min_days_to_expiry_exit": plan.min_days_to_expiry,
    }


def exit_plan_from_position(position: Any) -> HybridExitConfig:
    fallback = DynamicExitPlan(-0.25, -0.45, 0.30, 0.80, 0.25, 0.18, 0.12, 0.05, 0.18, 0.50, 0.22, 1.5)
    plan = DynamicExitPlan(
        safe_float(position.get("dynamic_stop_loss_pct", fallback.stop_loss_pct), fallback.stop_loss_pct),
        safe_float(position.get("dynamic_emergency_stop_loss_pct", fallback.emergency_stop_loss_pct), fallback.emergency_stop_loss_pct),
        safe_float(position.get("dynamic_soft_take_profit_pct", fallback.soft_take_profit_pct), fallback.soft_take_profit_pct),
        safe_float(position.get("dynamic_hard_take_profit_pct", fallback.hard_take_profit_pct), fallback.hard_take_profit_pct),
        safe_float(position.get("dynamic_trailing_activation_pct", fallback.trailing_activation_pct), fallback.trailing_activation_pct),
        safe_float(position.get("dynamic_trailing_drawdown_pct", fallback.trailing_drawdown_pct), fallback.trailing_drawdown_pct),
        safe_float(position.get("dynamic_trend_loss_profit_floor_pct", fallback.trend_loss_profit_floor_pct), fallback.trend_loss_profit_floor_pct),
        safe_float(position.get("dynamic_strong_trend_against_profit_floor_pct", fallback.strong_trend_against_profit_floor_pct), fallback.strong_trend_against_profit_floor_pct),
        safe_float(position.get("dynamic_volatility_contraction_profit_floor_pct", fallback.volatility_contraction_profit_floor_pct), fallback.volatility_contraction_profit_floor_pct),
        safe_float(position.get("dynamic_big_profit_giveback_activation_pct", fallback.big_profit_giveback_activation_pct), fallback.big_profit_giveback_activation_pct),
        safe_float(position.get("dynamic_big_profit_min_keep_pct", fallback.big_profit_min_keep_pct), fallback.big_profit_min_keep_pct),
        safe_float(position.get("dynamic_min_days_to_expiry_exit", fallback.min_days_to_expiry), fallback.min_days_to_expiry),
    )
    return plan.to_hybrid_config()
