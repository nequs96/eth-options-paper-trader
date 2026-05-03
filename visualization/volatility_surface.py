"""
visualization/volatility_surface.py

Interactive 3D implied volatility surface for ETH options.

This module uses real Deribit ETH option-chain data from:

    outputs/live_eth_option_chain.csv

It creates:
- CALL implied volatility surface
- PUT implied volatility surface
- combined 3D scatter plot

Outputs:
- outputs/eth_call_volatility_surface.html
- outputs/eth_put_volatility_surface.html
- outputs/eth_volatility_surface_scatter.html

Important:
This is for research and education only.
It does not place trades.
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


OPTION_CHAIN_FILE = "outputs/live_eth_option_chain.csv"
OUTPUT_FOLDER = "outputs"


def ensure_output_folder(folder: str = OUTPUT_FOLDER) -> None:
    """
    Create output folder if it does not exist.
    """

    Path(folder).mkdir(parents=True, exist_ok=True)


def load_live_option_chain(
    file_path: str = OPTION_CHAIN_FILE,
) -> pd.DataFrame:
    """
    Load live ETH option chain from CSV.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Option chain file not found: {file_path}. "
            "Run: python -m data.options_data"
        )

    data = pd.read_csv(path)

    if data.empty:
        raise ValueError("Option chain file is empty.")

    required_columns = {
        "instrument_name",
        "option_type",
        "strike",
        "days_to_expiry",
        "mark_iv",
    }

    missing_columns = required_columns.difference(set(data.columns))

    if missing_columns:
        raise ValueError(f"Option chain missing columns: {missing_columns}")

    data["strike"] = pd.to_numeric(data["strike"], errors="coerce")
    data["days_to_expiry"] = pd.to_numeric(data["days_to_expiry"], errors="coerce")
    data["mark_iv"] = pd.to_numeric(data["mark_iv"], errors="coerce")

    if "underlying_price" in data.columns:
        data["underlying_price"] = pd.to_numeric(
            data["underlying_price"],
            errors="coerce",
        )

    data = data.dropna(
        subset=[
            "option_type",
            "strike",
            "days_to_expiry",
            "mark_iv",
        ]
    ).copy()

    data = data[
        (data["strike"] > 0)
        & (data["days_to_expiry"] > 0)
        & (data["mark_iv"] > 0)
    ].copy()

    if data.empty:
        raise ValueError("No valid option-chain rows after cleaning.")

    data["option_type"] = data["option_type"].astype(str).str.lower().str.strip()

    return data.reset_index(drop=True)


def prepare_surface_data(
    option_chain: pd.DataFrame,
    option_type: str,
) -> tuple[list[float], list[float], list[list[float]]]:
    """
    Prepare strike / expiry / IV grid for a Plotly surface.

    Returns:
        strikes, days_to_expiry, iv_grid
    """

    option_type = option_type.lower().strip()

    filtered = option_chain[option_chain["option_type"] == option_type].copy()

    if filtered.empty:
        raise ValueError(f"No option data found for option_type={option_type}")

    pivot = filtered.pivot_table(
        index="days_to_expiry",
        columns="strike",
        values="mark_iv",
        aggfunc="mean",
    )

    pivot = pivot.sort_index(axis=0).sort_index(axis=1)

    # Fill missing values to make the surface smoother.
    pivot = pivot.interpolate(axis=0, limit_direction="both")
    pivot = pivot.interpolate(axis=1, limit_direction="both")

    pivot = pivot.dropna(axis=0, how="all")
    pivot = pivot.dropna(axis=1, how="all")

    if pivot.empty:
        raise ValueError(f"Could not build IV surface for {option_type} options.")

    strikes = [float(value) for value in pivot.columns.tolist()]
    days = [float(value) for value in pivot.index.tolist()]

    iv_grid = pivot.astype(float).values.tolist()

    return strikes, days, iv_grid


def plot_volatility_surface(
    option_chain: pd.DataFrame,
    option_type: str,
    save_path: str,
    show_plot: bool = True,
) -> go.Figure:
    """
    Plot 3D implied volatility surface for calls or puts.
    """

    strikes, days, iv_grid = prepare_surface_data(
        option_chain=option_chain,
        option_type=option_type,
    )

    title = f"ETH {option_type.upper()} Implied Volatility Surface"

    fig = go.Figure(
        data=[
            go.Surface(
                x=strikes,
                y=days,
                z=iv_grid,
                colorscale="Viridis",
                colorbar={
                    "title": "Mark IV",
                },
                hovertemplate=(
                    "Strike: %{x}<br>"
                    "Days to expiry: %{y:.2f}<br>"
                    "Mark IV: %{z:.2%}<extra></extra>"
                ),
            )
        ]
    )

    fig.update_layout(
        title=title,
        scene={
            "xaxis_title": "Strike price",
            "yaxis_title": "Days to expiry",
            "zaxis_title": "Implied volatility",
        },
        width=1000,
        height=750,
    )

    fig.write_html(save_path)

    if show_plot:
        fig.show()

    return fig


def plot_combined_volatility_scatter(
    option_chain: pd.DataFrame,
    save_path: str = "outputs/eth_volatility_surface_scatter.html",
    show_plot: bool = True,
) -> go.Figure:
    """
    Plot combined CALL and PUT implied volatility points as 3D scatter.

    This is useful when the option chain is sparse and a smooth surface may hide gaps.
    """

    plot_data = option_chain.dropna(
        subset=[
            "strike",
            "days_to_expiry",
            "mark_iv",
            "option_type",
        ]
    ).copy()

    if plot_data.empty:
        raise ValueError("No data available for combined volatility scatter.")

    fig = go.Figure()

    color_map = {
        "call": "green",
        "put": "red",
    }

    for option_type, group in plot_data.groupby("option_type"):
        strikes = pd.to_numeric(group["strike"], errors="coerce").astype(float).tolist()
        days = pd.to_numeric(
            group["days_to_expiry"],
            errors="coerce",
        ).astype(float).tolist()
        iv_values = pd.to_numeric(
            group["mark_iv"],
            errors="coerce",
        ).astype(float).tolist()

        instrument_names = group["instrument_name"].astype(str).tolist()

        fig.add_trace(
            go.Scatter3d(
                x=strikes,
                y=days,
                z=iv_values,
                mode="markers",
                name=str(option_type).upper(),
                marker={
                    "size": 5,
                    "color": color_map.get(str(option_type), "gray"),
                    "opacity": 0.75,
                },
                text=instrument_names,
                hovertemplate=(
                    "Instrument: %{text}<br>"
                    "Strike: %{x}<br>"
                    "Days to expiry: %{y:.2f}<br>"
                    "Mark IV: %{z:.2%}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="ETH Options Implied Volatility 3D Scatter",
        scene={
            "xaxis_title": "Strike price",
            "yaxis_title": "Days to expiry",
            "zaxis_title": "Implied volatility",
        },
        width=1000,
        height=750,
    )

    fig.write_html(save_path)

    if show_plot:
        fig.show()

    return fig


def print_surface_summary(option_chain: pd.DataFrame) -> None:
    """
    Print quick summary of volatility surface data.
    """

    print("\n========== ETH VOLATILITY SURFACE SUMMARY ==========")
    print(f"Rows:              {len(option_chain)}")

    if "underlying_price" in option_chain.columns:
        underlying = pd.to_numeric(
            option_chain["underlying_price"],
            errors="coerce",
        ).dropna()

        if not underlying.empty:
            print(f"ETH price:         ${float(underlying.iloc[0]):,.2f}")

    print(f"Option types:      {sorted(option_chain['option_type'].unique().tolist())}")
    print(f"Min strike:        ${option_chain['strike'].min():,.2f}")
    print(f"Max strike:        ${option_chain['strike'].max():,.2f}")
    print(f"Min DTE:           {option_chain['days_to_expiry'].min():.2f} days")
    print(f"Max DTE:           {option_chain['days_to_expiry'].max():.2f} days")
    print(f"Median mark IV:    {option_chain['mark_iv'].median():.2%}")
    print(f"Min mark IV:       {option_chain['mark_iv'].min():.2%}")
    print(f"Max mark IV:       {option_chain['mark_iv'].max():.2%}")
    print("====================================================\n")


def generate_volatility_surface_report(
    option_chain_file: str = OPTION_CHAIN_FILE,
    show_plot: bool = True,
) -> None:
    """
    Generate all volatility surface charts.
    """

    ensure_output_folder()

    option_chain = load_live_option_chain(option_chain_file)

    print_surface_summary(option_chain)

    option_types = set(option_chain["option_type"].unique().tolist())

    if "call" in option_types:
        plot_volatility_surface(
            option_chain=option_chain,
            option_type="call",
            save_path="outputs/eth_call_volatility_surface.html",
            show_plot=show_plot,
        )
        print("Saved: outputs/eth_call_volatility_surface.html")

    if "put" in option_types:
        plot_volatility_surface(
            option_chain=option_chain,
            option_type="put",
            save_path="outputs/eth_put_volatility_surface.html",
            show_plot=show_plot,
        )
        print("Saved: outputs/eth_put_volatility_surface.html")

    plot_combined_volatility_scatter(
        option_chain=option_chain,
        save_path="outputs/eth_volatility_surface_scatter.html",
        show_plot=show_plot,
    )

    print("Saved: outputs/eth_volatility_surface_scatter.html")

    print("\n========== VOLATILITY SURFACE REPORT GENERATED ==========")
    print("Open these files in your browser:")
    print("outputs/eth_call_volatility_surface.html")
    print("outputs/eth_put_volatility_surface.html")
    print("outputs/eth_volatility_surface_scatter.html")
    print("=========================================================")


if __name__ == "__main__":
    generate_volatility_surface_report(
        option_chain_file=OPTION_CHAIN_FILE,
        show_plot=True,
    )
