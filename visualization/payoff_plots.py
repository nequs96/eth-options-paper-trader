"""
visualization/payoff_plots.py

Payoff and profit/loss visualizations for ETH options.

This module creates charts for:
- long call payoff
- long put payoff
- profit/loss after premium
- strike price
- break-even point

These charts help visualize how an ETH option behaves at expiry.

Important:
This is for research and education. It is not financial advice.
"""

import numpy as np
import matplotlib.pyplot as plt


def generate_price_range(
    min_price: float,
    max_price: float,
    steps: int = 200,
) -> np.ndarray:
    """
    Generate a range of possible ETH prices at option expiry.

    Parameters
    ----------
    min_price : float
        Minimum ETH price.
    max_price : float
        Maximum ETH price.
    steps : int
        Number of points in the price range.

    Returns
    -------
    np.ndarray
        Array of ETH prices.
    """

    if min_price <= 0:
        raise ValueError("min_price must be greater than 0.")

    if max_price <= min_price:
        raise ValueError("max_price must be greater than min_price.")

    if steps <= 1:
        raise ValueError("steps must be greater than 1.")

    return np.linspace(min_price, max_price, steps)


def call_payoff(
    expiry_prices: np.ndarray,
    strike_price: float,
) -> np.ndarray:
    """
    Calculate long call payoff at expiry.

    Long call payoff:
        max(S - K, 0)

    Parameters
    ----------
    expiry_prices : np.ndarray
        Possible ETH prices at expiry.
    strike_price : float
        Option strike price.

    Returns
    -------
    np.ndarray
        Call payoff values.
    """

    if strike_price <= 0:
        raise ValueError("strike_price must be greater than 0.")

    return np.maximum(expiry_prices - strike_price, 0.0)


def put_payoff(
    expiry_prices: np.ndarray,
    strike_price: float,
) -> np.ndarray:
    """
    Calculate long put payoff at expiry.

    Long put payoff:
        max(K - S, 0)

    Parameters
    ----------
    expiry_prices : np.ndarray
        Possible ETH prices at expiry.
    strike_price : float
        Option strike price.

    Returns
    -------
    np.ndarray
        Put payoff values.
    """

    if strike_price <= 0:
        raise ValueError("strike_price must be greater than 0.")

    return np.maximum(strike_price - expiry_prices, 0.0)


def option_profit_loss(
    payoff: np.ndarray,
    premium_paid: float,
) -> np.ndarray:
    """
    Calculate option profit/loss after premium.

    Profit/loss:
        payoff - premium paid

    Parameters
    ----------
    payoff : np.ndarray
        Option payoff values.
    premium_paid : float
        Option premium paid.

    Returns
    -------
    np.ndarray
        Profit/loss values.
    """

    if premium_paid < 0:
        raise ValueError("premium_paid cannot be negative.")

    return payoff - premium_paid


def calculate_break_even(
    strike_price: float,
    premium_paid: float,
    option_type: str = "call",
) -> float:
    """
    Calculate option break-even price at expiry.

    Long call break-even:
        strike + premium

    Long put break-even:
        strike - premium
    """

    if strike_price <= 0:
        raise ValueError("strike_price must be greater than 0.")

    if premium_paid < 0:
        raise ValueError("premium_paid cannot be negative.")

    option_type = option_type.lower().strip()

    if option_type == "call":
        return float(strike_price + premium_paid)

    if option_type == "put":
        return float(strike_price - premium_paid)

    raise ValueError("option_type must be either 'call' or 'put'.")


def plot_option_payoff(
    strike_price: float,
    premium_paid: float,
    option_type: str = "call",
    min_price: float = 1000.0,
    max_price: float = 6000.0,
    steps: int = 300,
    show_plot: bool = True,
    save_path: str | None = None,
) -> None:
    """
    Plot payoff and profit/loss for a long ETH option.

    Parameters
    ----------
    strike_price : float
        Option strike price.
    premium_paid : float
        Premium paid for the option.
    option_type : str
        "call" or "put".
    min_price : float
        Minimum ETH expiry price shown on chart.
    max_price : float
        Maximum ETH expiry price shown on chart.
    steps : int
        Number of price points.
    show_plot : bool
        Whether to display the plot.
    save_path : str | None
        Optional file path to save chart image.
    """

    option_type = option_type.lower().strip()

    expiry_prices = generate_price_range(
        min_price=min_price,
        max_price=max_price,
        steps=steps,
    )

    if option_type == "call":
        payoff = call_payoff(expiry_prices, strike_price)
        title = "Long ETH Call Option Payoff"
    elif option_type == "put":
        payoff = put_payoff(expiry_prices, strike_price)
        title = "Long ETH Put Option Payoff"
    else:
        raise ValueError("option_type must be either 'call' or 'put'.")

    profit_loss = option_profit_loss(payoff, premium_paid)
    break_even = calculate_break_even(strike_price, premium_paid, option_type)

    plt.figure(figsize=(12, 7))

    plt.plot(
        expiry_prices,
        payoff,
        label="Payoff before premium",
        linewidth=2,
    )

    plt.plot(
        expiry_prices,
        profit_loss,
        label="Profit/Loss after premium",
        linewidth=2,
    )

    plt.axhline(
        0,
        color="black",
        linewidth=1,
        linestyle="--",
        label="Break-even P/L line",
    )

    plt.axvline(
        strike_price,
        color="orange",
        linewidth=1.5,
        linestyle="--",
        label=f"Strike price: ${strike_price:,.2f}",
    )

    plt.axvline(
        break_even,
        color="green",
        linewidth=1.5,
        linestyle="--",
        label=f"Break-even: ${break_even:,.2f}",
    )

    plt.title(title)
    plt.xlabel("ETH price at expiry")
    plt.ylabel("Payoff / Profit-Loss")
    plt.legend()
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_call_and_put_payoffs(
    strike_price: float,
    call_premium: float,
    put_premium: float,
    min_price: float = 1000.0,
    max_price: float = 6000.0,
    steps: int = 300,
    show_plot: bool = True,
    save_path: str | None = None,
) -> None:
    """
    Plot long call and long put profit/loss on the same chart.

    This is useful for comparing how call and put options behave.
    """

    expiry_prices = generate_price_range(
        min_price=min_price,
        max_price=max_price,
        steps=steps,
    )

    call_pl = option_profit_loss(
        call_payoff(expiry_prices, strike_price),
        call_premium,
    )

    put_pl = option_profit_loss(
        put_payoff(expiry_prices, strike_price),
        put_premium,
    )

    call_break_even = calculate_break_even(
        strike_price=strike_price,
        premium_paid=call_premium,
        option_type="call",
    )

    put_break_even = calculate_break_even(
        strike_price=strike_price,
        premium_paid=put_premium,
        option_type="put",
    )

    plt.figure(figsize=(12, 7))

    plt.plot(
        expiry_prices,
        call_pl,
        label=f"Long call P/L, premium ${call_premium:,.2f}",
        linewidth=2,
    )

    plt.plot(
        expiry_prices,
        put_pl,
        label=f"Long put P/L, premium ${put_premium:,.2f}",
        linewidth=2,
    )

    plt.axhline(
        0,
        color="black",
        linewidth=1,
        linestyle="--",
    )

    plt.axvline(
        strike_price,
        color="orange",
        linewidth=1.5,
        linestyle="--",
        label=f"Strike: ${strike_price:,.2f}",
    )

    plt.axvline(
        call_break_even,
        color="green",
        linewidth=1.2,
        linestyle=":",
        label=f"Call break-even: ${call_break_even:,.2f}",
    )

    plt.axvline(
        put_break_even,
        color="red",
        linewidth=1.2,
        linestyle=":",
        label=f"Put break-even: ${put_break_even:,.2f}",
    )

    plt.title("ETH Long Call vs Long Put Profit/Loss")
    plt.xlabel("ETH price at expiry")
    plt.ylabel("Profit / Loss")
    plt.legend()
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    # Simple standalone test

    strike = 3200.0
    call_premium = 175.0
    put_premium = 365.0

    plot_option_payoff(
        strike_price=strike,
        premium_paid=call_premium,
        option_type="call",
        min_price=1000.0,
        max_price=6000.0,
    )

    plot_option_payoff(
        strike_price=strike,
        premium_paid=put_premium,
        option_type="put",
        min_price=1000.0,
        max_price=6000.0,
    )

    plot_call_and_put_payoffs(
        strike_price=strike,
        call_premium=call_premium,
        put_premium=put_premium,
        min_price=1000.0,
        max_price=6000.0,
    )