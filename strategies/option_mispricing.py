"""
strategies/option_mispricing.py

Basic ETH options mispricing research signal.

This module compares:
- market option price
- Black-Scholes theoretical price
- implied volatility
- historical volatility

The goal is to classify an option as:
- expensive
- cheap
- neutral

Important:
This is a research signal, not financial advice.
This file does not place trades.
"""

from dataclasses import dataclass

from models.black_scholes import black_scholes_price
from models.implied_volatility import (
    implied_volatility,
    volatility_spread,
    volatility_spread_percent,
)


@dataclass
class MispricingResult:
    """
    Container for option mispricing analysis.
    """

    option_type: str
    spot_price: float
    strike_price: float
    time_to_expiry: float
    risk_free_rate: float
    market_price: float
    theoretical_price: float
    price_difference: float
    price_difference_percent: float
    implied_volatility: float
    historical_volatility: float
    volatility_spread: float
    volatility_spread_percent_points: float
    classification: str
    signal: str
    explanation: str


def calculate_price_difference(
    market_price: float,
    theoretical_price: float,
) -> float:
    """
    Calculate absolute difference between market price and theoretical price.

    Positive result:
        market price is above theoretical price

    Negative result:
        market price is below theoretical price
    """

    return float(market_price - theoretical_price)


def calculate_price_difference_percent(
    market_price: float,
    theoretical_price: float,
) -> float:
    """
    Calculate percentage difference between market price and theoretical price.

    Formula:
        (market_price - theoretical_price) / theoretical_price
    """

    if theoretical_price <= 0:
        raise ValueError("theoretical_price must be greater than 0.")

    difference = calculate_price_difference(
        market_price=market_price,
        theoretical_price=theoretical_price,
    )

    return float(difference / theoretical_price)


def classify_option_mispricing(
    price_difference_percent: float,
    vol_spread: float,
    price_threshold: float = 0.10,
    volatility_threshold: float = 0.10,
) -> str:
    """
    Classify option as expensive, cheap, or neutral.

    Parameters
    ----------
    price_difference_percent : float
        Market price premium/discount vs theoretical price.
    vol_spread : float
        Implied volatility minus historical volatility.
    price_threshold : float
        Price mispricing threshold.
        Example: 0.10 means 10%.
    volatility_threshold : float
        Volatility spread threshold.
        Example: 0.10 means 10 volatility points.

    Returns
    -------
    str
        "expensive", "cheap", or "neutral"
    """

    if price_threshold < 0:
        raise ValueError("price_threshold cannot be negative.")

    if volatility_threshold < 0:
        raise ValueError("volatility_threshold cannot be negative.")

    expensive_by_price = price_difference_percent > price_threshold
    expensive_by_vol = vol_spread > volatility_threshold

    cheap_by_price = price_difference_percent < -price_threshold
    cheap_by_vol = vol_spread < -volatility_threshold

    if expensive_by_price and expensive_by_vol:
        return "expensive"

    if cheap_by_price and cheap_by_vol:
        return "cheap"

    return "neutral"


def generate_research_signal(
    classification: str,
    option_type: str,
) -> str:
    """
    Generate a simple research signal from classification.

    This is not an execution signal.
    It only describes what the model sees.
    """

    classification = classification.lower().strip()
    option_type = option_type.lower().strip()

    if classification == "expensive":
        return f"{option_type}_looks_expensive"

    if classification == "cheap":
        return f"{option_type}_looks_cheap"

    return "no_clear_edge"


def build_explanation(
    classification: str,
    price_difference_percent: float,
    vol_spread_percent_points: float,
) -> str:
    """
    Build human-readable explanation for the signal.
    """

    price_diff_pct = price_difference_percent * 100.0

    if classification == "expensive":
        return (
            "The option appears expensive because the market price is "
            f"{price_diff_pct:.2f}% above theoretical value and implied volatility "
            f"is {vol_spread_percent_points:.2f} percentage points above historical volatility."
        )

    if classification == "cheap":
        return (
            "The option appears cheap because the market price is "
            f"{abs(price_diff_pct):.2f}% below theoretical value and implied volatility "
            f"is {abs(vol_spread_percent_points):.2f} percentage points below historical volatility."
        )

    return (
        "No strong mispricing detected. Price difference and volatility spread "
        "do not both pass the selected thresholds."
    )


def analyze_option_mispricing(
    market_price: float,
    spot_price: float,
    strike_price: float,
    time_to_expiry: float,
    risk_free_rate: float,
    historical_volatility: float,
    option_type: str = "call",
    price_threshold: float = 0.10,
    volatility_threshold: float = 0.10,
) -> MispricingResult:
    """
    Analyze option mispricing using Black-Scholes and implied volatility.

    Parameters
    ----------
    market_price : float
        Observed market option price.
    spot_price : float
        Current ETH spot price.
    strike_price : float
        Option strike price.
    time_to_expiry : float
        Time to expiry in years.
    risk_free_rate : float
        Annual risk-free rate.
    historical_volatility : float
        Historical volatility as decimal.
    option_type : str
        "call" or "put".
    price_threshold : float
        Price mispricing threshold.
    volatility_threshold : float
        Volatility spread threshold.

    Returns
    -------
    MispricingResult
        Full mispricing analysis.
    """

    option_type = option_type.lower().strip()

    if market_price <= 0:
        raise ValueError("market_price must be greater than 0.")

    if spot_price <= 0:
        raise ValueError("spot_price must be greater than 0.")

    if strike_price <= 0:
        raise ValueError("strike_price must be greater than 0.")

    if time_to_expiry <= 0:
        raise ValueError("time_to_expiry must be greater than 0.")

    if historical_volatility <= 0:
        raise ValueError("historical_volatility must be greater than 0.")

    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be either 'call' or 'put'.")

    theoretical_price = black_scholes_price(
        S=spot_price,
        K=strike_price,
        T=time_to_expiry,
        r=risk_free_rate,
        sigma=historical_volatility,
        option_type=option_type,
    )

    iv = implied_volatility(
        market_price=market_price,
        S=spot_price,
        K=strike_price,
        T=time_to_expiry,
        r=risk_free_rate,
        option_type=option_type,
    )

    price_difference = calculate_price_difference(
        market_price=market_price,
        theoretical_price=theoretical_price,
    )

    price_difference_percent = calculate_price_difference_percent(
        market_price=market_price,
        theoretical_price=theoretical_price,
    )

    vol_spread = volatility_spread(
        implied_vol=iv,
        historical_vol=historical_volatility,
    )

    vol_spread_percent_points = volatility_spread_percent(
        implied_vol=iv,
        historical_vol=historical_volatility,
    )

    classification = classify_option_mispricing(
        price_difference_percent=price_difference_percent,
        vol_spread=vol_spread,
        price_threshold=price_threshold,
        volatility_threshold=volatility_threshold,
    )

    signal = generate_research_signal(
        classification=classification,
        option_type=option_type,
    )

    explanation = build_explanation(
        classification=classification,
        price_difference_percent=price_difference_percent,
        vol_spread_percent_points=vol_spread_percent_points,
    )

    return MispricingResult(
        option_type=option_type,
        spot_price=float(spot_price),
        strike_price=float(strike_price),
        time_to_expiry=float(time_to_expiry),
        risk_free_rate=float(risk_free_rate),
        market_price=float(market_price),
        theoretical_price=float(theoretical_price),
        price_difference=float(price_difference),
        price_difference_percent=float(price_difference_percent),
        implied_volatility=float(iv),
        historical_volatility=float(historical_volatility),
        volatility_spread=float(vol_spread),
        volatility_spread_percent_points=float(vol_spread_percent_points),
        classification=classification,
        signal=signal,
        explanation=explanation,
    )


def print_mispricing_result(result: MispricingResult) -> None:
    """
    Print mispricing result in a clean format.
    """

    print("========== OPTION MISPRICING ANALYSIS ==========")
    print(f"Option type:                 {result.option_type.upper()}")
    print(f"ETH spot price:              ${result.spot_price:,.2f}")
    print(f"Strike price:                ${result.strike_price:,.2f}")
    print(f"Market option price:         ${result.market_price:,.4f}")
    print(f"Theoretical option price:    ${result.theoretical_price:,.4f}")
    print("------------------------------------------------")
    print(f"Price difference:            ${result.price_difference:,.4f}")
    print(f"Price difference percent:    {result.price_difference_percent:.2%}")
    print(f"Implied volatility:          {result.implied_volatility:.2%}")
    print(f"Historical volatility:       {result.historical_volatility:.2%}")
    print(f"Volatility spread:           {result.volatility_spread:.2%}")
    print(
        f"Vol spread percentage pts:   "
        f"{result.volatility_spread_percent_points:.2f}"
    )
    print("------------------------------------------------")
    print(f"Classification:              {result.classification.upper()}")
    print(f"Research signal:             {result.signal}")
    print(f"Explanation:                 {result.explanation}")
    print("================================================")


if __name__ == "__main__":
    # Standalone test with simulated ETH option market price.

    S = 3000.0
    K = 3200.0
    T = 30 / 365
    r = 0.04
    historical_vol = 0.75
    option_type = "call"

    # Simulated market price.
    # Later this should come from a real options exchange.
    market_price = 220.0

    result = analyze_option_mispricing(
        market_price=market_price,
        spot_price=S,
        strike_price=K,
        time_to_expiry=T,
        risk_free_rate=r,
        historical_volatility=historical_vol,
        option_type=option_type,
        price_threshold=0.10,
        volatility_threshold=0.10,
    )

    print_mispricing_result(result)