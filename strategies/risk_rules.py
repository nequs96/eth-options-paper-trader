"""
strategies/risk_rules.py

Risk management rules for ETH options strategies.

This module enforces:
- max risk per trade
- max position size
- volatility sanity checks
- capital protection rules

This file DOES NOT place trades.
It only approves or rejects proposed trades.
"""

from dataclasses import dataclass


@dataclass
class RiskDecision:
    allowed: bool
    position_size: float
    risk_amount: float
    reason: str


def calculate_position_size(
    capital: float,
    max_risk_per_trade: float,
    option_price: float,
) -> float:
    """
    Calculate position size based on risk per trade.

    Parameters
    ----------
    capital : float
        Total capital.
    max_risk_per_trade : float
        Max fraction of capital to risk (e.g. 0.01 = 1%).
    option_price : float
        Option premium (max loss for long option).

    Returns
    -------
    float
        Number of option contracts (can be fractional for research).
    """

    if capital <= 0:
        raise ValueError("capital must be greater than 0.")

    if max_risk_per_trade <= 0 or max_risk_per_trade > 1:
        raise ValueError("max_risk_per_trade must be between 0 and 1.")

    if option_price <= 0:
        raise ValueError("option_price must be greater than 0.")

    risk_amount = capital * max_risk_per_trade
    position_size = risk_amount / option_price

    return float(position_size)


def approve_trade(
    capital: float,
    option_price: float,
    implied_volatility: float,
    historical_volatility: float,
    max_risk_per_trade: float = 0.01,
    max_volatility: float = 2.0,
    min_volatility: float = 0.10,
) -> RiskDecision:
    """
    Decide whether a trade is allowed under risk rules.

    Parameters
    ----------
    capital : float
        Total capital.
    option_price : float
        Option premium (max loss).
    implied_volatility : float
        Implied volatility as decimal.
    historical_volatility : float
        Historical volatility as decimal.
    max_risk_per_trade : float
        Max capital risk per trade.
    max_volatility : float
        Absolute volatility ceiling (panic filter).
    min_volatility : float
        Absolute volatility floor (illiquid/no-move filter).

    Returns
    -------
    RiskDecision
        Approval decision with explanation.
    """

    if implied_volatility <= 0 or historical_volatility <= 0:
        return RiskDecision(
            allowed=False,
            position_size=0.0,
            risk_amount=0.0,
            reason="Invalid volatility values.",
        )

    if implied_volatility > max_volatility:
        return RiskDecision(
            allowed=False,
            position_size=0.0,
            risk_amount=0.0,
            reason="Implied volatility too high (panic regime).",
        )

    if implied_volatility < min_volatility:
        return RiskDecision(
            allowed=False,
            position_size=0.0,
            risk_amount=0.0,
            reason="Implied volatility too low (no movement expected).",
        )

    position_size = calculate_position_size(
        capital=capital,
        max_risk_per_trade=max_risk_per_trade,
        option_price=option_price,
    )

    risk_amount = position_size * option_price

    if position_size <= 0:
        return RiskDecision(
            allowed=False,
            position_size=0.0,
            risk_amount=0.0,
            reason="Calculated position size is zero.",
        )

    return RiskDecision(
        allowed=True,
        position_size=float(position_size),
        risk_amount=float(risk_amount),
        reason="Trade approved under risk rules.",
    )


def print_risk_decision(decision: RiskDecision) -> None:
    """
    Print risk decision in readable format.
    """

    print("========== RISK DECISION ==========")
    print(f"Trade allowed:     {decision.allowed}")
    print(f"Position size:    {decision.position_size:.4f}")
    print(f"Capital at risk:  ${decision.risk_amount:,.2f}")
    print(f"Reason:           {decision.reason}")
    print("===================================")


if __name__ == "__main__":
    # Standalone test

    capital = 10_000.0
    option_price = 200.0
    implied_vol = 0.85
    historical_vol = 0.70

    decision = approve_trade(
        capital=capital,
        option_price=option_price,
        implied_volatility=implied_vol,
        historical_volatility=historical_vol,
        max_risk_per_trade=0.01,
    )

    print_risk_decision(decision)   