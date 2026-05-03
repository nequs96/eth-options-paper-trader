"""
models/black_scholes.py

Black-Scholes pricing model for European call and put options.

This module is used to calculate theoretical ETH option prices.

Important:
Black-Scholes is a pricing model, not a guaranteed trading strategy.
For crypto options, the model is only a baseline because ETH has:
- high volatility
- jumps
- changing implied volatility
- liquidity issues
- 24/7 trading
"""

import math
from scipy.stats import norm


def _validate_inputs(S: float, K: float, T: float, r: float, sigma: float) -> None:
    """
    Validate Black-Scholes inputs.

    Parameters
    ----------
    S : float
        Current price of the underlying asset.
    K : float
        Strike price.
    T : float
        Time to expiry in years.
    r : float
        Annual risk-free rate as decimal.
    sigma : float
        Annualized volatility as decimal.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """

    if S <= 0:
        raise ValueError("Underlying price S must be greater than 0.")

    if K <= 0:
        raise ValueError("Strike price K must be greater than 0.")

    if T < 0:
        raise ValueError("Time to expiry T cannot be negative.")

    if sigma < 0:
        raise ValueError("Volatility sigma cannot be negative.")


def calculate_d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate d1 used in the Black-Scholes formula.

    d1 = [ln(S/K) + (r + 0.5*sigma^2)*T] / [sigma*sqrt(T)]
    """

    _validate_inputs(S, K, T, r, sigma)

    if T == 0 or sigma == 0:
        return float("inf") if S > K else float("-inf")

    return (
        math.log(S / K) + (r + 0.5 * sigma ** 2) * T
    ) / (sigma * math.sqrt(T))


def calculate_d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate d2 used in the Black-Scholes formula.

    d2 = d1 - sigma*sqrt(T)
    """

    _validate_inputs(S, K, T, r, sigma)

    if T == 0 or sigma == 0:
        return float("inf") if S > K else float("-inf")

    d1 = calculate_d1(S, K, T, r, sigma)
    return d1 - sigma * math.sqrt(T)


def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate the theoretical price of a European call option.
    """

    _validate_inputs(S, K, T, r, sigma)

    if T == 0:
        return float(max(S - K, 0.0))

    if sigma == 0:
        forward_value = S - K * math.exp(-r * T)
        return float(max(forward_value, 0.0))

    d1 = calculate_d1(S, K, T, r, sigma)
    d2 = calculate_d2(S, K, T, r, sigma)

    call_price = S * float(norm.cdf(d1)) - K * math.exp(-r * T) * float(norm.cdf(d2))

    return float(max(call_price, 0.0))

def black_scholes_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate the theoretical price of a European put option.
    """

    _validate_inputs(S, K, T, r, sigma)

    if T == 0:
        return float(max(K - S, 0.0))

    if sigma == 0:
        forward_value = K * math.exp(-r * T) - S
        return float(max(forward_value, 0.0))

    d1 = calculate_d1(S, K, T, r, sigma)
    d2 = calculate_d2(S, K, T, r, sigma)

    put_price = K * math.exp(-r * T) * float(norm.cdf(-d2)) - S * float(norm.cdf(-d1))

    return float(max(put_price, 0.0))


def black_scholes_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> float:
    """
    General Black-Scholes pricing function.

    Parameters
    ----------
    S : float
        Current underlying price.
    K : float
        Strike price.
    T : float
        Time to expiry in years.
    r : float
        Risk-free rate.
    sigma : float
        Annualized volatility.
    option_type : str
        Either "call" or "put".

    Returns
    -------
    float
        Theoretical option price.
    """

    option_type = option_type.lower().strip()

    if option_type == "call":
        return black_scholes_call(S, K, T, r, sigma)

    if option_type == "put":
        return black_scholes_put(S, K, T, r, sigma)

    raise ValueError("option_type must be either 'call' or 'put'.")


def intrinsic_value(S: float, K: float, option_type: str) -> float:
    """
    Calculate intrinsic value of an option.

    Call intrinsic value:
        max(S - K, 0)

    Put intrinsic value:
        max(K - S, 0)
    """

    if S <= 0:
        raise ValueError("Underlying price S must be greater than 0.")

    if K <= 0:
        raise ValueError("Strike price K must be greater than 0.")

    option_type = option_type.lower().strip()

    if option_type == "call":
        return max(S - K, 0.0)

    if option_type == "put":
        return max(K - S, 0.0)

    raise ValueError("option_type must be either 'call' or 'put'.")


def time_value(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
) -> float:
    """
    Calculate option time value.

    Time value = option price - intrinsic value
    """

    price = black_scholes_price(S, K, T, r, sigma, option_type)
    intrinsic = intrinsic_value(S, K, option_type)

    return max(price - intrinsic, 0.0)


if __name__ == "__main__":
    # Simple test example for ETH option pricing.

    S = 3000.0        # Current ETH price
    K = 3200.0        # Strike price
    T = 30 / 365      # 30 days to expiry
    r = 0.04          # 4% risk-free rate
    sigma = 0.75      # 75% annualized volatility

    call_price = black_scholes_call(S, K, T, r, sigma)
    put_price = black_scholes_put(S, K, T, r, sigma)

    print("========== Black-Scholes ETH Option Pricing ==========")
    print(f"ETH price: ${S:,.2f}")
    print(f"Strike price: ${K:,.2f}")
    print(f"Time to expiry: {T:.4f} years")
    print(f"Risk-free rate: {r:.2%}")
    print(f"Volatility: {sigma:.2%}")
    print("------------------------------------------------------")
    print(f"Call price: ${call_price:,.2f}")
    print(f"Put price: ${put_price:,.2f}")
    print("======================================================")