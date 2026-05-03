"""
models/implied_volatility.py

Implied volatility solver for ETH options.

This module calculates implied volatility from a market option price.

Black-Scholes normally works like this:

    spot price + strike + expiry + rate + volatility -> option price

Implied volatility reverses that:

    spot price + strike + expiry + rate + market option price -> volatility

This is useful because options markets often express expectations through
implied volatility.

Important:
This module is for research and education. It is not financial advice.
"""

from typing import cast

from scipy.optimize import brentq

from models.black_scholes import black_scholes_price, intrinsic_value


def _validate_implied_volatility_inputs(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str,
) -> None:
    """
    Validate inputs for implied volatility calculation.
    """

    if market_price <= 0:
        raise ValueError("market_price must be greater than 0.")

    if S <= 0:
        raise ValueError("Underlying price S must be greater than 0.")

    if K <= 0:
        raise ValueError("Strike price K must be greater than 0.")

    if T <= 0:
        raise ValueError("Time to expiry T must be greater than 0.")

    option_type = option_type.lower().strip()

    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be either 'call' or 'put'.")

    intrinsic = intrinsic_value(
        S=S,
        K=K,
        option_type=option_type,
    )

    if market_price < intrinsic:
        raise ValueError(
            "market_price is below intrinsic value. "
            "This usually means the option price is invalid, stale, "
            "or affected by bad market data."
        )


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    min_vol: float = 0.0001,
    max_vol: float = 5.0,
) -> float:
    """
    Calculate implied volatility using numerical root finding.

    Parameters
    ----------
    market_price : float
        Observed market price of the option.
    S : float
        Current underlying price, for example ETH spot price.
    K : float
        Option strike price.
    T : float
        Time to expiry in years.
    r : float
        Annual risk-free rate as decimal.
    option_type : str
        "call" or "put".
    min_vol : float
        Minimum volatility search bound.
    max_vol : float
        Maximum volatility search bound.

    Returns
    -------
    float
        Implied volatility as decimal.

    Example:
        0.75 means 75% annualized implied volatility.
    """

    option_type = option_type.lower().strip()

    _validate_implied_volatility_inputs(
        market_price=market_price,
        S=S,
        K=K,
        T=T,
        r=r,
        option_type=option_type,
    )

    if min_vol <= 0:
        raise ValueError("min_vol must be greater than 0.")

    if max_vol <= min_vol:
        raise ValueError("max_vol must be greater than min_vol.")

    def pricing_error(volatility: float) -> float:
        """
        Difference between Black-Scholes theoretical price and market price.

        Root is found when:

            theoretical_price - market_price = 0
        """

        theoretical_price = black_scholes_price(
            S=S,
            K=K,
            T=T,
            r=r,
            sigma=volatility,
            option_type=option_type,
        )

        return float(theoretical_price - market_price)

    low_error = pricing_error(min_vol)
    high_error = pricing_error(max_vol)

    if low_error * high_error > 0:
        raise ValueError(
            "Could not solve implied volatility within bounds. "
            "Try increasing max_vol or check whether market_price is realistic."
        )

    # scipy.optimize.brentq returns a float at runtime.
    # Pylance may incorrectly infer a tuple because brentq has multiple overloads.
    # cast(float, ...) makes the type explicit for Pylance.
    root = brentq(
        pricing_error,
        min_vol,
        max_vol,
        maxiter=100,
        xtol=1e-8,
    )

    iv = cast(float, root)

    return float(iv)


def implied_volatility_percent(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
) -> float:
    """
    Calculate implied volatility and return it as percentage.

    Example:
        returns 75.0 instead of 0.75
    """

    iv = implied_volatility(
        market_price=market_price,
        S=S,
        K=K,
        T=T,
        r=r,
        option_type=option_type,
    )

    return float(iv * 100.0)


def volatility_spread(
    implied_vol: float,
    historical_vol: float,
) -> float:
    """
    Calculate spread between implied volatility and historical volatility.

    Formula:
        spread = implied_volatility - historical_volatility

    Positive spread:
        implied volatility is higher than historical volatility.

    Negative spread:
        implied volatility is lower than historical volatility.
    """

    if implied_vol < 0:
        raise ValueError("implied_vol cannot be negative.")

    if historical_vol < 0:
        raise ValueError("historical_vol cannot be negative.")

    return float(implied_vol - historical_vol)


def volatility_spread_percent(
    implied_vol: float,
    historical_vol: float,
) -> float:
    """
    Calculate volatility spread in percentage points.

    Example:
        implied_vol = 0.90
        historical_vol = 0.70

        result = 20.0
    """

    spread = volatility_spread(
        implied_vol=implied_vol,
        historical_vol=historical_vol,
    )

    return float(spread * 100.0)


def classify_volatility_mispricing(
    implied_vol: float,
    historical_vol: float,
    threshold: float = 0.10,
) -> str:
    """
    Classify option volatility as expensive, cheap, or neutral.

    Parameters
    ----------
    implied_vol : float
        Implied volatility as decimal.
    historical_vol : float
        Historical volatility as decimal.
    threshold : float
        Difference threshold as decimal.

    Returns
    -------
    str
        "expensive", "cheap", or "neutral"
    """

    if threshold < 0:
        raise ValueError("threshold cannot be negative.")

    spread = volatility_spread(
        implied_vol=implied_vol,
        historical_vol=historical_vol,
    )

    if spread > threshold:
        return "expensive"

    if spread < -threshold:
        return "cheap"

    return "neutral"


def implied_volatility_summary(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    historical_vol: float,
    option_type: str = "call",
    threshold: float = 0.10,
) -> dict[str, float | str]:
    """
    Create a full implied volatility summary.

    Returns:
    - market option price
    - implied volatility
    - historical volatility
    - volatility spread
    - volatility spread in percentage points
    - mispricing classification
    """

    iv = implied_volatility(
        market_price=market_price,
        S=S,
        K=K,
        T=T,
        r=r,
        option_type=option_type,
    )

    spread = volatility_spread(
        implied_vol=iv,
        historical_vol=historical_vol,
    )

    classification = classify_volatility_mispricing(
        implied_vol=iv,
        historical_vol=historical_vol,
        threshold=threshold,
    )

    return {
        "market_price": float(market_price),
        "implied_volatility": float(iv),
        "implied_volatility_percent": float(iv * 100.0),
        "historical_volatility": float(historical_vol),
        "historical_volatility_percent": float(historical_vol * 100.0),
        "volatility_spread": float(spread),
        "volatility_spread_percent_points": float(spread * 100.0),
        "classification": classification,
    }


if __name__ == "__main__":
    # Standalone test example.

    S = 3000.0
    K = 3200.0
    T = 30 / 365
    r = 0.04
    option_type = "call"

    # Example market price.
    # Later this should come from a real options exchange like Deribit.
    market_price = 200.0

    # Example historical volatility.
    historical_vol = 0.75

    summary = implied_volatility_summary(
        market_price=market_price,
        S=S,
        K=K,
        T=T,
        r=r,
        historical_vol=historical_vol,
        option_type=option_type,
        threshold=0.10,
    )

    print("========== IMPLIED VOLATILITY TEST ==========")
    print(f"Option type: {option_type.upper()}")
    print(f"ETH spot price: ${S:,.2f}")
    print(f"Strike price: ${K:,.2f}")
    print(f"Market option price: ${market_price:,.2f}")
    print("---------------------------------------------")

    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key:<35}: {value:.6f}")
        else:
            print(f"{key:<35}: {value}")

    print("=============================================")