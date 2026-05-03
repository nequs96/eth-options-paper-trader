"""
visualization/surface_plots.py

3D surface visualizations for ETH options using the Black-Scholes model.

This module creates:
- option price surface
- delta surface
- gamma surface
- vega surface
- theta surface

The most important first chart is the option price surface:

X-axis: ETH spot price
Y-axis: days to expiry
Z-axis: theoretical option price

Important:
These visualizations are for research and education.
They are not financial advice.
"""

import numpy as np
import plotly.graph_objects as go

from models.black_scholes import black_scholes_price
from models.greeks import delta, gamma, vega, theta


def create_price_expiry_grid(
    min_price: float,
    max_price: float,
    price_steps: int,
    min_days: int,
    max_days: int,
    expiry_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a grid of ETH prices and days to expiry.

    Parameters
    ----------
    min_price : float
        Minimum ETH spot price.
    max_price : float
        Maximum ETH spot price.
    price_steps : int
        Number of ETH price points.
    min_days : int
        Minimum days to expiry.
    max_days : int
        Maximum days to expiry.
    expiry_steps : int
        Number of expiry points.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        spot_prices, days_to_expiry, spot_grid, days_grid
    """

    if min_price <= 0:
        raise ValueError("min_price must be greater than 0.")

    if max_price <= min_price:
        raise ValueError("max_price must be greater than min_price.")

    if price_steps <= 1:
        raise ValueError("price_steps must be greater than 1.")

    if min_days <= 0:
        raise ValueError("min_days must be greater than 0.")

    if max_days <= min_days:
        raise ValueError("max_days must be greater than min_days.")

    if expiry_steps <= 1:
        raise ValueError("expiry_steps must be greater than 1.")

    spot_prices = np.linspace(min_price, max_price, price_steps)
    days_to_expiry = np.linspace(min_days, max_days, expiry_steps)

    spot_grid, days_grid = np.meshgrid(spot_prices, days_to_expiry)

    return spot_prices, days_to_expiry, spot_grid, days_grid


def calculate_option_price_surface(
    spot_grid: np.ndarray,
    days_grid: np.ndarray,
    strike_price: float,
    risk_free_rate: float,
    volatility: float,
    option_type: str = "call",
    days_in_year: float = 365.0,
) -> np.ndarray:
    """
    Calculate a Black-Scholes option price surface.

    Parameters
    ----------
    spot_grid : np.ndarray
        Grid of ETH spot prices.
    days_grid : np.ndarray
        Grid of days to expiry.
    strike_price : float
        Option strike price.
    risk_free_rate : float
        Annual risk-free rate as decimal.
    volatility : float
        Annualized volatility as decimal.
    option_type : str
        "call" or "put".
    days_in_year : float
        Number of days in one year.

    Returns
    -------
    np.ndarray
        Option price surface.
    """

    if strike_price <= 0:
        raise ValueError("strike_price must be greater than 0.")

    if volatility <= 0:
        raise ValueError("volatility must be greater than 0.")

    option_type = option_type.lower().strip()

    surface = np.zeros_like(spot_grid, dtype=float)

    rows, cols = spot_grid.shape

    for row in range(rows):
        for col in range(cols):
            S = float(spot_grid[row, col])
            T = float(days_grid[row, col]) / days_in_year

            surface[row, col] = black_scholes_price(
                S=S,
                K=strike_price,
                T=T,
                r=risk_free_rate,
                sigma=volatility,
                option_type=option_type,
            )

    return surface


def calculate_greek_surface(
    greek_name: str,
    spot_grid: np.ndarray,
    days_grid: np.ndarray,
    strike_price: float,
    risk_free_rate: float,
    volatility: float,
    option_type: str = "call",
    days_in_year: float = 365.0,
) -> np.ndarray:
    """
    Calculate a Greek surface.

    Available Greek names:
    - "delta"
    - "gamma"
    - "vega"
    - "theta"

    Parameters
    ----------
    greek_name : str
        Name of Greek to calculate.
    spot_grid : np.ndarray
        Grid of ETH spot prices.
    days_grid : np.ndarray
        Grid of days to expiry.
    strike_price : float
        Option strike price.
    risk_free_rate : float
        Annual risk-free rate.
    volatility : float
        Annualized volatility.
    option_type : str
        "call" or "put".
    days_in_year : float
        Number of days in one year.

    Returns
    -------
    np.ndarray
        Greek value surface.
    """

    greek_name = greek_name.lower().strip()
    option_type = option_type.lower().strip()

    if greek_name not in {"delta", "gamma", "vega", "theta"}:
        raise ValueError("greek_name must be one of: delta, gamma, vega, theta.")

    surface = np.zeros_like(spot_grid, dtype=float)

    rows, cols = spot_grid.shape

    for row in range(rows):
        for col in range(cols):
            S = float(spot_grid[row, col])
            T = float(days_grid[row, col]) / days_in_year

            if greek_name == "delta":
                surface[row, col] = delta(
                    S=S,
                    K=strike_price,
                    T=T,
                    r=risk_free_rate,
                    sigma=volatility,
                    option_type=option_type,
                )

            elif greek_name == "gamma":
                surface[row, col] = gamma(
                    S=S,
                    K=strike_price,
                    T=T,
                    r=risk_free_rate,
                    sigma=volatility,
                )

            elif greek_name == "vega":
                surface[row, col] = vega(
                    S=S,
                    K=strike_price,
                    T=T,
                    r=risk_free_rate,
                    sigma=volatility,
                )

            elif greek_name == "theta":
                surface[row, col] = theta(
                    S=S,
                    K=strike_price,
                    T=T,
                    r=risk_free_rate,
                    sigma=volatility,
                    option_type=option_type,
                )

    return surface


def plot_option_price_surface(
    strike_price: float,
    risk_free_rate: float,
    volatility: float,
    option_type: str = "call",
    min_price: float = 1000.0,
    max_price: float = 6000.0,
    price_steps: int = 80,
    min_days: int = 1,
    max_days: int = 180,
    expiry_steps: int = 60,
    days_in_year: float = 365.0,
    show_plot: bool = True,
) -> go.Figure:
    """
    Plot a 3D Black-Scholes option price surface.

    Returns
    -------
    plotly.graph_objects.Figure
        Plotly figure object.
    """

    _, _, spot_grid, days_grid = create_price_expiry_grid(
        min_price=min_price,
        max_price=max_price,
        price_steps=price_steps,
        min_days=min_days,
        max_days=max_days,
        expiry_steps=expiry_steps,
    )

    price_surface = calculate_option_price_surface(
        spot_grid=spot_grid,
        days_grid=days_grid,
        strike_price=strike_price,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        option_type=option_type,
        days_in_year=days_in_year,
    )

    title = (
        f"ETH {option_type.upper()} Option Price Surface "
        f"| Strike ${strike_price:,.0f}, Vol {volatility:.0%}"
    )

    fig = go.Figure(
        data=[
            go.Surface(
                x=spot_grid,
                y=days_grid,
                z=price_surface,
                colorscale="Viridis",
                colorbar=dict(title="Option Price"),
            )
        ]
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="ETH Spot Price",
            yaxis_title="Days to Expiry",
            zaxis_title="Option Price",
        ),
        width=950,
        height=700,
    )

    if show_plot:
        fig.show()

    return fig


def plot_greek_surface(
    greek_name: str,
    strike_price: float,
    risk_free_rate: float,
    volatility: float,
    option_type: str = "call",
    min_price: float = 1000.0,
    max_price: float = 6000.0,
    price_steps: int = 80,
    min_days: int = 1,
    max_days: int = 180,
    expiry_steps: int = 60,
    days_in_year: float = 365.0,
    show_plot: bool = True,
) -> go.Figure:
    """
    Plot a 3D Greek surface.

    Available Greek names:
    - delta
    - gamma
    - vega
    - theta
    """

    greek_name = greek_name.lower().strip()

    _, _, spot_grid, days_grid = create_price_expiry_grid(
        min_price=min_price,
        max_price=max_price,
        price_steps=price_steps,
        min_days=min_days,
        max_days=max_days,
        expiry_steps=expiry_steps,
    )

    greek_surface = calculate_greek_surface(
        greek_name=greek_name,
        spot_grid=spot_grid,
        days_grid=days_grid,
        strike_price=strike_price,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        option_type=option_type,
        days_in_year=days_in_year,
    )

    title = (
        f"ETH {option_type.upper()} {greek_name.capitalize()} Surface "
        f"| Strike ${strike_price:,.0f}, Vol {volatility:.0%}"
    )

    fig = go.Figure(
        data=[
            go.Surface(
                x=spot_grid,
                y=days_grid,
                z=greek_surface,
                colorscale="Plasma",
                colorbar=dict(title=greek_name.capitalize()),
            )
        ]
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="ETH Spot Price",
            yaxis_title="Days to Expiry",
            zaxis_title=greek_name.capitalize(),
        ),
        width=950,
        height=700,
    )

    if show_plot:
        fig.show()

    return fig


if __name__ == "__main__":
    # Standalone test

    plot_option_price_surface(
        strike_price=3200.0,
        risk_free_rate=0.04,
        volatility=0.75,
        option_type="call",
        min_price=1000.0,
        max_price=6000.0,
        price_steps=80,
        min_days=1,
        max_days=180,
        expiry_steps=60,
        show_plot=True,
    )

    plot_greek_surface(
        greek_name="delta",
        strike_price=3200.0,
        risk_free_rate=0.04,
        volatility=0.75,
        option_type="call",
        min_price=1000.0,
        max_price=6000.0,
        price_steps=80,
        min_days=1,
        max_days=180,
        expiry_steps=60,
        show_plot=True,
    )