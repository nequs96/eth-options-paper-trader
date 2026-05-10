"""Dynamic risk sizing for ETH option paper trades."""
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from models.market_confidence import MarketConfidence, safe_float, clamp


@dataclass(frozen=True)
class DynamicRiskDecision:
    allowed: bool
    risk_pct: float
    risk_amount_usd: float
    quantity: float
    reason: str
    confidence_bucket: str
    drawdown_multiplier: float
    liquidity_multiplier: float
    portfolio_multiplier: float


@dataclass(frozen=True)
class DynamicRiskConfig:
    min_risk_per_trade: float = 0.001
    normal_max_risk_per_trade: float = 0.0125
    exceptional_max_risk_per_trade: float = 0.020
    max_total_open_risk_pct: float = 0.10
    min_mci_to_trade: float = 0.35
    min_cash_buffer_pct: float = 0.05
    max_drawdown_for_new_trades: float = -0.08


def drawdown_multiplier(current_drawdown: float) -> float:
    dd = min(float(current_drawdown), 0.0)
    if dd <= -0.08:
        return 0.0
    if dd <= -0.06:
        return 0.25
    if dd <= -0.04:
        return 0.50
    if dd <= -0.02:
        return 0.75
    return 1.0


def confidence_bucket_and_cap(mci: float, cfg: DynamicRiskConfig) -> tuple[str, float]:
    if mci < 0.35:
        return "reject", 0.0
    if mci < 0.55:
        return "probe", 0.0035
    if mci < 0.70:
        return "normal", 0.0075
    if mci < 0.85:
        return "strong", cfg.normal_max_risk_per_trade
    return "exceptional", cfg.exceptional_max_risk_per_trade


def calculate_dynamic_risk_pct(confidence: MarketConfidence, current_drawdown: float, open_risk_pct: float, cfg: DynamicRiskConfig) -> tuple[float, str, float, float, float]:
    bucket, bucket_cap = confidence_bucket_and_cap(confidence.mci, cfg)
    if bucket == "reject":
        return 0.0, bucket, 0.0, 0.0, 0.0

    base = cfg.min_risk_per_trade
    raw = base + (cfg.normal_max_risk_per_trade - base) * (confidence.mci ** 2.25)
    if bucket == "exceptional" and confidence.liquidity_score >= 0.80 and confidence.vol_score >= 0.70 and current_drawdown > -0.02:
        raw = base + (cfg.exceptional_max_risk_per_trade - base) * (confidence.mci ** 2.25)

    dd_mult = drawdown_multiplier(current_drawdown)
    liq_mult = clamp(confidence.liquidity_score ** 1.5, 0.15, 1.0)
    portfolio_capacity = clamp((cfg.max_total_open_risk_pct - open_risk_pct) / max(cfg.max_total_open_risk_pct, 1e-9), 0.0, 1.0)
    portfolio_mult = clamp(0.35 + 0.65 * portfolio_capacity, 0.0, 1.0)
    risk_pct = min(raw * dd_mult * liq_mult * portfolio_mult, bucket_cap)
    return float(max(0.0, risk_pct)), bucket, float(dd_mult), float(liq_mult), float(portfolio_mult)


def calculate_open_risk_pct(open_positions: pd.DataFrame, initial_cash: float) -> float:
    if open_positions.empty or initial_cash <= 0:
        return 0.0
    data = open_positions.copy()
    if "status" in data.columns:
        data = data[data["status"].astype(str).str.lower().eq("open")]
    if data.empty or "capital_at_risk" not in data.columns:
        return 0.0
    risk = pd.to_numeric(data["capital_at_risk"], errors="coerce").fillna(0.0).sum()
    return float(risk / initial_cash)


def size_position(
    cash: float,
    initial_cash: float,
    option_price_usd: float,
    confidence: MarketConfidence,
    current_drawdown: float,
    open_risk_pct: float,
    cfg: DynamicRiskConfig | None = None,
) -> DynamicRiskDecision:
    if cfg is None:
        cfg = DynamicRiskConfig()
    price = safe_float(option_price_usd)
    if price <= 0:
        return DynamicRiskDecision(False, 0.0, 0.0, 0.0, "invalid_option_price", "reject", 0.0, 0.0, 0.0)
    if cash <= initial_cash * cfg.min_cash_buffer_pct:
        return DynamicRiskDecision(False, 0.0, 0.0, 0.0, "cash_buffer_reached", "reject", 0.0, 0.0, 0.0)
    if current_drawdown <= cfg.max_drawdown_for_new_trades:
        return DynamicRiskDecision(False, 0.0, 0.0, 0.0, "max_drawdown_block", "reject", 0.0, 0.0, 0.0)
    if confidence.reject_reason:
        return DynamicRiskDecision(False, 0.0, 0.0, 0.0, confidence.reject_reason, "reject", 0.0, 0.0, 0.0)

    risk_pct, bucket, dd_mult, liq_mult, port_mult = calculate_dynamic_risk_pct(confidence, current_drawdown, open_risk_pct, cfg)
    if risk_pct <= 0:
        return DynamicRiskDecision(False, 0.0, 0.0, 0.0, "risk_pct_zero", bucket, dd_mult, liq_mult, port_mult)

    max_total_risk_remaining = max(cfg.max_total_open_risk_pct - open_risk_pct, 0.0) * initial_cash
    risk_amount = min(initial_cash * risk_pct, cash, max_total_risk_remaining)
    quantity = risk_amount / price if price > 0 else 0.0
    if risk_amount <= 0 or quantity <= 0:
        return DynamicRiskDecision(False, 0.0, 0.0, 0.0, "no_remaining_risk_capacity", bucket, dd_mult, liq_mult, port_mult)

    return DynamicRiskDecision(True, float(risk_amount / initial_cash), float(risk_amount), float(quantity), "accepted_dynamic_risk", bucket, dd_mult, liq_mult, port_mult)
