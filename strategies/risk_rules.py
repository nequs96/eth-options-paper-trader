from __future__ import annotations
from dataclasses import dataclass

@dataclass
class RiskDecision:
    allowed: bool
    position_size: float
    risk_amount: float
    reason: str
    max_loss_allowed: float = 0.0
    position_notional: float = 0.0
    risk_reward: float = 0.0
    daily_drawdown: float = 0.0


def calculate_daily_drawdown(starting_day_equity: float, current_equity: float) -> float:
    return 0.0 if starting_day_equity <= 0 else current_equity / starting_day_equity - 1.0


def calculate_risk_reward(entry_price: float, stop_loss: float, take_profit: float) -> float:
    risk = abs(entry_price - stop_loss)
    return 0.0 if risk == 0 else abs(take_profit - entry_price) / risk


def calculate_position_size(capital: float, max_risk_per_trade: float, option_price: float, contract_multiplier: float = 1.0, allow_fractional_size: bool = True) -> float:
    if capital <= 0 or option_price <= 0 or contract_multiplier <= 0:
        return 0.0
    qty = (capital * max_risk_per_trade) / (option_price * contract_multiplier)
    return qty if allow_fractional_size else float(int(qty))


def approve_trade(capital: float, option_price: float, implied_volatility: float, historical_volatility: float, max_risk_per_trade: float = 0.01, **kwargs) -> RiskDecision:
    q = calculate_position_size(capital, max_risk_per_trade, option_price)
    return RiskDecision(q > 0, q, q * option_price, 'approved' if q > 0 else 'zero_size')


def print_risk_decision(decision: RiskDecision) -> None:
    print(decision)
