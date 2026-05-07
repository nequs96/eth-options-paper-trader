"""
strategies/risk_rules.py

Risk management rules for ETH options strategies.

This module enforces:
- max risk per trade
- max position size
- daily drawdown protection
- volatility sanity checks
- risk/reward requirements
- capital protection rules
- safe validation against invalid market data

This file DOES NOT place trades.
It only approves or rejects proposed trades.
"""

from dataclasses import dataclass
import math


# =========================
# Default risk configuration
# =========================

DEFAULT_MAX_RISK_PER_TRADE = 0.01       # 1% of capital at risk per trade
DEFAULT_MAX_POSITION_PCT = 0.10         # max 10% of capital in one position
DEFAULT_MAX_DAILY_DRAWDOWN = 0.02       # stop trading after 2% daily drawdown
DEFAULT_MIN_RISK_REWARD = 1.50          # minimum reward/risk ratio

DEFAULT_MAX_VOLATILITY = 2.00           # 200% IV panic filter
DEFAULT_MIN_VOLATILITY = 0.10           # 10% IV low-volatility filter
DEFAULT_MAX_IV_HV_RATIO = 3.00          # IV cannot be more than 3x HV
DEFAULT_MIN_IV_HV_RATIO = 0.50          # IV cannot be less than 0.5x HV

DEFAULT_CONTRACT_MULTIPLIER = 1.0       # for crypto options this may be 1
DEFAULT_ALLOW_FRACTIONAL_SIZE = True    # useful for research/backtests

MIN_OPTION_PRICE = 0.0001               # protects against divide-by-zero
MIN_CAPITAL = 1.0                       # avoids meaningless tiny capital


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


def _is_valid_number(value: float) -> bool:
    """
    Return True only for normal finite numbers.
    Rejects None, NaN, inf, strings, etc.
    """
    return isinstance(value, (int, float)) and math.isfinite(value)


def _reject(reason: str) -> RiskDecision:
    """
    Helper for consistent rejection responses.
    """
    return RiskDecision(
        allowed=False,
        position_size=0.0,
        risk_amount=0.0,
        reason=reason,
    )


def calculate_daily_drawdown(
    starting_day_equity: float,
    current_equity: float,
) -> float:
    """
    Calculate current daily drawdown as a decimal.

    Example:
    starting_day_equity = 10000
    current_equity = 9800
    drawdown = 0.02
    """

    if not _is_valid_number(starting_day_equity) or starting_day_equity <= 0:
        return 1.0

    if not _is_valid_number(current_equity) or current_equity <= 0:
        return 1.0

    drawdown = (starting_day_equity - current_equity) / starting_day_equity

    # If equity is above starting equity, drawdown should not be negative.
    return max(0.0, float(drawdown))


def calculate_risk_reward(
    entry_price: float,
    stop_loss: float,
    take_profit: float,
) -> float:
    """
    Calculate reward/risk ratio.

    For a long option:
    - entry_price = premium paid
    - stop_loss = price where you exit for loss
    - take_profit = price where you exit for profit

    Example:
    entry = 200
    stop = 150
    target = 300

    risk = 50
    reward = 100
    risk_reward = 2.0
    """

    values = [entry_price, stop_loss, take_profit]

    if not all(_is_valid_number(v) for v in values):
        return 0.0

    if entry_price <= 0 or stop_loss < 0 or take_profit <= 0:
        return 0.0

    risk = abs(entry_price - stop_loss)
    reward = abs(take_profit - entry_price)

    if risk <= 0:
        return 0.0

    return float(reward / risk)


def calculate_position_size(
    capital: float,
    max_risk_per_trade: float,
    option_price: float,
    contract_multiplier: float = DEFAULT_CONTRACT_MULTIPLIER,
    allow_fractional_size: bool = DEFAULT_ALLOW_FRACTIONAL_SIZE,
) -> float:
    """
    Calculate position size based on max risk per trade.

    For long options, maximum loss is usually the premium paid.

    Parameters
    ----------
    capital : float
        Total account capital/equity.
    max_risk_per_trade : float
        Max fraction of capital to risk.
        Example: 0.01 = 1%.
    option_price : float
        Option premium.
    contract_multiplier : float
        Contract multiplier.
        For some crypto options this may be 1.
        For stock options this is often 100.
    allow_fractional_size : bool
        If True, allows fractional position size for research/backtesting.
        If False, floors size to whole contracts.

    Returns
    -------
    float
        Number of contracts/units.
    """

    if not _is_valid_number(capital) or capital < MIN_CAPITAL:
        raise ValueError("capital must be a valid number greater than minimum capital.")

    if not _is_valid_number(max_risk_per_trade):
        raise ValueError("max_risk_per_trade must be a valid number.")

    if max_risk_per_trade <= 0 or max_risk_per_trade > 1:
        raise ValueError("max_risk_per_trade must be between 0 and 1.")

    if not _is_valid_number(option_price) or option_price < MIN_OPTION_PRICE:
        raise ValueError("option_price must be a valid positive number.")

    if not _is_valid_number(contract_multiplier) or contract_multiplier <= 0:
        raise ValueError("contract_multiplier must be greater than 0.")

    risk_amount = capital * max_risk_per_trade
    risk_per_contract = option_price * contract_multiplier

    if risk_per_contract <= 0:
        raise ValueError("risk_per_contract must be greater than 0.")

    position_size = risk_amount / risk_per_contract

    if not allow_fractional_size:
        position_size = math.floor(position_size)

    return float(position_size)


def approve_trade(
    capital: float,
    option_price: float,
    implied_volatility: float,
    historical_volatility: float,
    max_risk_per_trade: float = DEFAULT_MAX_RISK_PER_TRADE,
    max_volatility: float = DEFAULT_MAX_VOLATILITY,
    min_volatility: float = DEFAULT_MIN_VOLATILITY,
    starting_day_equity: float | None = None,
    current_equity: float | None = None,
    max_daily_drawdown: float = DEFAULT_MAX_DAILY_DRAWDOWN,
    max_position_pct: float = DEFAULT_MAX_POSITION_PCT,
    entry_price: float | None = None,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    min_risk_reward: float = DEFAULT_MIN_RISK_REWARD,
    contract_multiplier: float = DEFAULT_CONTRACT_MULTIPLIER,
    allow_fractional_size: bool = DEFAULT_ALLOW_FRACTIONAL_SIZE,
    max_iv_hv_ratio: float = DEFAULT_MAX_IV_HV_RATIO,
    min_iv_hv_ratio: float = DEFAULT_MIN_IV_HV_RATIO,
) -> RiskDecision:
    """
    Decide whether a proposed trade is allowed under risk rules.

    Parameters
    ----------
    capital : float
        Total account capital/equity.
    option_price : float
        Option premium.
    implied_volatility : float
        Implied volatility as decimal.
        Example: 0.85 = 85%.
    historical_volatility : float
        Historical volatility as decimal.
        Example: 0.70 = 70%.
    max_risk_per_trade : float
        Max fraction of account to risk per trade.
    max_volatility : float
        Absolute IV ceiling.
    min_volatility : float
        Absolute IV floor.
    starting_day_equity : float | None
        Equity at start of day. Used for daily drawdown kill-switch.
    current_equity : float | None
        Current equity. Used for daily drawdown kill-switch.
    max_daily_drawdown : float
        Max allowed daily drawdown.
    max_position_pct : float
        Max fraction of capital allowed in one position.
    entry_price : float | None
        Trade entry price. If None, option_price is used.
    stop_loss : float | None
        Planned stop loss price.
    take_profit : float | None
        Planned take profit price.
    min_risk_reward : float
        Minimum acceptable reward/risk ratio.
    contract_multiplier : float
        Option contract multiplier.
    allow_fractional_size : bool
        If False, position size is floored to whole contracts.
    max_iv_hv_ratio : float
        Maximum allowed IV/HV ratio.
    min_iv_hv_ratio : float
        Minimum allowed IV/HV ratio.

    Returns
    -------
    RiskDecision
        Approval/rejection decision with explanation.
    """

    # =========================
    # Basic input validation
    # =========================

    if not _is_valid_number(capital) or capital < MIN_CAPITAL:
        return _reject("Invalid capital.")

    if not _is_valid_number(option_price) or option_price < MIN_OPTION_PRICE:
        return _reject("Invalid option price.")

    if not _is_valid_number(implied_volatility) or implied_volatility <= 0:
        return _reject("Invalid implied volatility.")

    if not _is_valid_number(historical_volatility) or historical_volatility <= 0:
        return _reject("Invalid historical volatility.")

    if not _is_valid_number(max_risk_per_trade) or max_risk_per_trade <= 0 or max_risk_per_trade > 1:
        return _reject("Invalid max risk per trade.")

    if not _is_valid_number(max_position_pct) or max_position_pct <= 0 or max_position_pct > 1:
        return _reject("Invalid max position percentage.")

    if not _is_valid_number(contract_multiplier) or contract_multiplier <= 0:
        return _reject("Invalid contract multiplier.")

    # =========================
    # Daily drawdown kill-switch
    # =========================

    daily_drawdown = 0.0

    if starting_day_equity is not None and current_equity is not None:
        daily_drawdown = calculate_daily_drawdown(
            starting_day_equity=starting_day_equity,
            current_equity=current_equity,
        )

        if daily_drawdown >= max_daily_drawdown:
            decision = _reject(
                f"Daily drawdown limit exceeded: {daily_drawdown:.2%} "
                f">= {max_daily_drawdown:.2%}."
            )
            decision.daily_drawdown = daily_drawdown
            return decision

    # =========================
    # Volatility filters
    # =========================

    if implied_volatility > max_volatility:
        return _reject(
            f"Implied volatility too high: {implied_volatility:.2%} "
            f"> {max_volatility:.2%}. Panic regime filter triggered."
        )

    if implied_volatility < min_volatility:
        return _reject(
            f"Implied volatility too low: {implied_volatility:.2%} "
            f"< {min_volatility:.2%}. Low-movement filter triggered."
        )

    iv_hv_ratio = implied_volatility / historical_volatility

    if iv_hv_ratio > max_iv_hv_ratio:
        return _reject(
            f"IV/HV ratio too high: {iv_hv_ratio:.2f}. "
            f"Options may be overpriced or market may be unstable."
        )

    if iv_hv_ratio < min_iv_hv_ratio:
        return _reject(
            f"IV/HV ratio too low: {iv_hv_ratio:.2f}. "
            f"Volatility data may be stale or unreliable."
        )

    # =========================
    # Position sizing
    # =========================

    try:
        position_size = calculate_position_size(
            capital=capital,
            max_risk_per_trade=max_risk_per_trade,
            option_price=option_price,
            contract_multiplier=contract_multiplier,
            allow_fractional_size=allow_fractional_size,
        )
    except ValueError as exc:
        return _reject(str(exc))

    if position_size <= 0:
        return _reject("Calculated position size is zero.")

    risk_amount = position_size * option_price * contract_multiplier
    max_loss_allowed = capital * max_risk_per_trade
    position_notional = position_size * option_price * contract_multiplier
    max_position_notional = capital * max_position_pct

    # This should usually be equal to max_loss_allowed for long options,
    # but keep the check for safety.
    if risk_amount > max_loss_allowed + 1e-9:
        return RiskDecision(
            allowed=False,
            position_size=0.0,
            risk_amount=0.0,
            reason=(
                f"Risk amount too high: ${risk_amount:,.2f} "
                f"> allowed ${max_loss_allowed:,.2f}."
            ),
            max_loss_allowed=max_loss_allowed,
            position_notional=position_notional,
            daily_drawdown=daily_drawdown,
        )

    if position_notional > max_position_notional:
        return RiskDecision(
            allowed=False,
            position_size=0.0,
            risk_amount=0.0,
            reason=(
                f"Position too large: ${position_notional:,.2f} "
                f"> max allowed ${max_position_notional:,.2f}."
            ),
            max_loss_allowed=max_loss_allowed,
            position_notional=position_notional,
            daily_drawdown=daily_drawdown,
        )

    # =========================
    # Risk/reward filter
    # =========================

    risk_reward = 0.0

    if entry_price is not None or stop_loss is not None or take_profit is not None:
        if entry_price is None:
            entry_price = option_price

        if stop_loss is None or take_profit is None:
            return _reject("Risk/reward check requested but stop_loss or take_profit is missing.")

        risk_reward = calculate_risk_reward(
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        if risk_reward <= 0:
            return _reject("Invalid risk/reward calculation.")

        if risk_reward < min_risk_reward:
            return RiskDecision(
                allowed=False,
                position_size=0.0,
                risk_amount=0.0,
                reason=(
                    f"Risk/reward too low: {risk_reward:.2f} "
                    f"< required {min_risk_reward:.2f}."
                ),
                max_loss_allowed=max_loss_allowed,
                position_notional=position_notional,
                risk_reward=risk_reward,
                daily_drawdown=daily_drawdown,
            )

    # =========================
    # Final approval
    # =========================

    return RiskDecision(
        allowed=True,
        position_size=float(position_size),
        risk_amount=float(risk_amount),
        reason="Trade approved under risk rules.",
        max_loss_allowed=float(max_loss_allowed),
        position_notional=float(position_notional),
        risk_reward=float(risk_reward),
        daily_drawdown=float(daily_drawdown),
    )


def print_risk_decision(decision: RiskDecision) -> None:
    """
    Print risk decision in readable format.
    """

    print("========== RISK DECISION ==========")
    print(f"Trade allowed:       {decision.allowed}")
    print(f"Position size:       {decision.position_size:.4f}")
    print(f"Capital at risk:     ${decision.risk_amount:,.2f}")
    print(f"Max loss allowed:    ${decision.max_loss_allowed:,.2f}")
    print(f"Position notional:   ${decision.position_notional:,.2f}")
    print(f"Risk/reward:         {decision.risk_reward:.2f}")
    print(f"Daily drawdown:      {decision.daily_drawdown:.2%}")
    print(f"Reason:              {decision.reason}")
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
        starting_day_equity=10_000.0,
        current_equity=9_950.0,
        entry_price=200.0,
        stop_loss=150.0,
        take_profit=300.0,
        contract_multiplier=1.0,
        allow_fractional_size=True,
    )

    print_risk_decision(decision)