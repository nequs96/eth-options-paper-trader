"""
models/greeks.py

Option Greeks for European call and put options using the Black-Scholes model.

Greeks measure how sensitive an option price is to changes in:
- underlying price
- volatility
- time
- interest rates

For ETH options, the most important Greeks are usually:
- Delta
- Gamma
- Vega
- Theta

Important:
These calculations are theoretical and based on Black-Scholes assumptions.
Crypto options can behave differently because ETH has high volatility,
jumps, liquidity issues, and changing implied volatility.
"""

import math
from scipy.stats import norm

from models.black_scholes import calculate_d1, calculate_d2


def _validate_inputs(S: float, K: float, T: float, r: float, sigma: float) -> None:
    """
    Validate inputs for Greek calculations.
    """

    if S <= 0:
        raise ValueError("Underlying price S must be greater than 0.")

    if K <= 0:
        raise ValueError("Strike price K must be greater than 0.")

    if T <= 0:
        raise ValueError("Time to expiry T must be greater than 0 for Greeks.")

    if sigma <= 0:
        raise ValueError("Volatility sigma must be greater than 0 for Greeks.")


def delta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> float:
    """
    Calculate option Delta.

    Delta measures how much the option price changes when the underlying price changes.

    Example:
    If call delta = 0.60, then a $1 increase in ETH theoretically increases
    the call option price by about $0.60.

    Parameters
    ----------
    S : float
        Current underlying price.
    K : float
        Strike price.
    T : float
        Time to expiry in years.
    r : float
        Risk-free rate as decimal.
    sigma : float
        Annualized volatility as decimal.
    option_type : str
        "call" or "put".

    Returns
    -------
    float
        Option Delta.
    """

    _validate_inputs(S, K, T, r, sigma)

    option_type = option_type.lower().strip()
    d1 = calculate_d1(S, K, T, r, sigma)

    if option_type == "call":
        return float(norm.cdf(d1))

    if option_type == "put":
        return float(norm.cdf(d1) - 1.0)

    raise ValueError("option_type must be either 'call' or 'put'.")


def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate option Gamma.

    Gamma measures how much Delta changes when the underlying price changes.

    High Gamma means the option Delta can change quickly.
    This is especially important for short-dated ETH options.
    """

    _validate_inputs(S, K, T, r, sigma)

    d1 = calculate_d1(S, K, T, r, sigma)

    gamma_value = float(norm.pdf(d1)) / (S * sigma * math.sqrt(T))

    return float(gamma_value)


def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate option Vega.

    Vega measures how much the option price changes when implied volatility changes.

    This function returns Vega per 1.00 volatility change.
    To get Vega per 1 percentage point change in volatility, divide by 100.

    Example:
    If vega = 350, then a volatility move from 75% to 76%
    changes the option price by about 350 / 100 = $3.50.
    """

    _validate_inputs(S, K, T, r, sigma)

    d1 = calculate_d1(S, K, T, r, sigma)

    vega_value = S * float(norm.pdf(d1)) * math.sqrt(T)

    return float(vega_value)


def vega_per_1_percent(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate Vega per 1 percentage point change in volatility.

    This is often easier to interpret than raw Vega.
    """

    return float(vega(S, K, T, r, sigma) / 100.0)


def theta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> float:
    """
    Calculate option Theta.

    Theta measures how much the option price changes as time passes.

    This function returns annual Theta.
    For daily Theta, use theta_per_day().

    Usually, long options have negative Theta because they lose time value.
    """

    _validate_inputs(S, K, T, r, sigma)

    option_type = option_type.lower().strip()

    d1 = calculate_d1(S, K, T, r, sigma)
    d2 = calculate_d2(S, K, T, r, sigma)

    first_term = -(S * float(norm.pdf(d1)) * sigma) / (2.0 * math.sqrt(T))

    if option_type == "call":
        second_term = r * K * math.exp(-r * T) * float(norm.cdf(d2))
        theta_value = first_term - second_term
        return float(theta_value)

    if option_type == "put":
        second_term = r * K * math.exp(-r * T) * float(norm.cdf(-d2))
        theta_value = first_term + second_term
        return float(theta_value)

    raise ValueError("option_type must be either 'call' or 'put'.")


def theta_per_day(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> float:
    """
    Calculate daily Theta.

    This estimates how much option value is lost or gained per day.
    """

    return float(theta(S, K, T, r, sigma, option_type) / 365.0)


def rho(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> float:
    """
    Calculate option Rho.

    Rho measures sensitivity to interest rate changes.

    For ETH options, Rho is usually less important than Delta, Gamma, Vega, and Theta.
    This function returns Rho per 1.00 interest rate change.
    To get Rho per 1 percentage point change, divide by 100.
    """

    _validate_inputs(S, K, T, r, sigma)

    option_type = option_type.lower().strip()

    d2 = calculate_d2(S, K, T, r, sigma)

    if option_type == "call":
        rho_value = K * T * math.exp(-r * T) * float(norm.cdf(d2))
        return float(rho_value)

    if option_type == "put":
        rho_value = -K * T * math.exp(-r * T) * float(norm.cdf(-d2))
        return float(rho_value)

    raise ValueError("option_type must be either 'call' or 'put'.")


def rho_per_1_percent(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> float:
    """
    Calculate Rho per 1 percentage point change in interest rates.
    """

    return float(rho(S, K, T, r, sigma, option_type) / 100.0)


def all_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> dict[str, float]:
    """
    Calculate all main Greeks and return them as a dictionary.
    """

    return {
        "delta": delta(S, K, T, r, sigma, option_type),
        "gamma": gamma(S, K, T, r, sigma),
        "vega": vega(S, K, T, r, sigma),
        "vega_per_1_percent": vega_per_1_percent(S, K, T, r, sigma),
        "theta": theta(S, K, T, r, sigma, option_type),
        "theta_per_day": theta_per_day(S, K, T, r, sigma, option_type),
        "rho": rho(S, K, T, r, sigma, option_type),
        "rho_per_1_percent": rho_per_1_percent(S, K, T, r, sigma, option_type),
    }


if __name__ == "__main__":
    # Simple test example for ETH option Greeks.

    S = 3000.0
    K = 3200.0
    T = 30 / 365
    r = 0.04
    sigma = 0.75

    option_type = "call"

    greeks = all_greeks(S, K, T, r, sigma, option_type)

    print("========== ETH Option Greeks ==========")
    print(f"Option type: {option_type}")
    print(f"ETH price: ${S:,.2f}")
    print(f"Strike price: ${K:,.2f}")
    print(f"Time to expiry: {T:.4f} years")
    print(f"Risk-free rate: {r:.2%}")
    print(f"Volatility: {sigma:.2%}")
    print("---------------------------------------")

    for greek_name, greek_value in greeks.items():
        print(f"{greek_name}: {greek_value:.6f}")

    print("=======================================")