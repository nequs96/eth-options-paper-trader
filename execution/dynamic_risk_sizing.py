from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from models.market_confidence import MarketConfidence, safe_float


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
    exceptional_max_risk_per_trade: float = 0.02
    max_total_open_risk_pct: float = 0.10
    min_mci_to_trade: float = 0.35
    min_cash_buffer_pct: float = 0.05
    max_drawdown_for_new_trades: float = -0.08


def calculate_open_risk_pct(open_positions: pd.DataFrame, initial_cash: float) -> float:
    if open_positions.empty or initial_cash <= 0 or 'capital_at_risk' not in open_positions.columns:
        return 0.0
    data = open_positions.copy()
    if 'status' in data.columns:
        data = data[data['status'].astype(str).str.lower().eq('open')]
    return float(pd.to_numeric(data['capital_at_risk'], errors='coerce').fillna(0.0).sum() / initial_cash)


def size_position(cash: float, initial_cash: float, option_price_usd: float, confidence: MarketConfidence, current_drawdown: float, open_risk_pct: float, cfg: DynamicRiskConfig | None = None) -> DynamicRiskDecision:
    cfg = cfg or DynamicRiskConfig()
    price = safe_float(option_price_usd)
    if price <= 0:
        return DynamicRiskDecision(False, 0, 0, 0, 'invalid_option_price', 'reject', 0, 0, 0)
    if confidence.reject_reason:
        return DynamicRiskDecision(False, 0, 0, 0, confidence.reject_reason, 'reject', 0, 0, 0)
    if confidence.mci < cfg.min_mci_to_trade:
        return DynamicRiskDecision(False, 0, 0, 0, 'mci_below_trade_minimum', 'reject', 0, 0, 0)
    risk_pct = min(cfg.normal_max_risk_per_trade, cfg.min_risk_per_trade + (cfg.normal_max_risk_per_trade - cfg.min_risk_per_trade) * confidence.mci)
    remaining_risk = max(cfg.max_total_open_risk_pct - open_risk_pct, 0.0) * initial_cash
    cash_buffer = cfg.min_cash_buffer_pct * initial_cash
    available_cash = max(cash - cash_buffer, 0.0)
    risk_amount = min(initial_cash * risk_pct, available_cash, remaining_risk)
    quantity = risk_amount / price if price > 0 else 0.0
    return DynamicRiskDecision(risk_amount > 0, risk_amount / initial_cash if initial_cash else 0.0, risk_amount, quantity, 'accepted_dynamic_risk', 'normal', 1.0, confidence.liquidity_score, 1.0)
