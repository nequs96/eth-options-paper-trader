"""
visualization/live_option_report.py

Visual report for live ETH option scanner and live paper-backtest.

This module reads:
- outputs/live_backtest_candidates.csv
- outputs/live_paper_positions.csv

And creates:
- market price vs model price chart
- implied volatility vs historical volatility chart
- volatility spread by strike
- classification count chart
- top candidate score chart
- paper positions report

Important:
This is for research and education only.
It does not place trades.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


CANDIDATES_FILE = "outputs/live_backtest_candidates.csv"
POSITIONS_FILE = "outputs/live_paper_positions.csv"
OUTPUT_FOLDER = "outputs"


def ensure_output_folder(folder: str = OUTPUT_FOLDER) -> None:
    """
    Create output folder if it does not exist.
    """

    Path(folder).mkdir(parents=True, exist_ok=True)


def load_candidates(
    file_path: str = CANDIDATES_FILE,
) -> pd.DataFrame:
    """
    Load live option candidates CSV.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Candidates file not found: {file_path}. "
            "Run: python -m backtesting.live_option_backtest_engine"
        )

    data = pd.read_csv(path)

    if data.empty:
        raise ValueError("Candidates file is empty.")

    numeric_columns = [
        "spot_price",
        "strike",
        "days_to_expiry",
        "market_price_usd",
        "model_price_usd",
        "price_diff_usd",
        "price_diff_pct",
        "implied_volatility",
        "historical_volatility",
        "volatility_spread",
        "cheapness_score",
        "volatility_edge",
        "combined_score",
        "bid_ask_spread_pct",
        "delta",
        "gamma",
        "theta",
        "vega",
        "open_interest",
    ]

    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    required_columns = {
        "instrument_name",
        "option_type",
        "strike",
        "market_price_usd",
        "model_price_usd",
        "implied_volatility",
        "historical_volatility",
        "volatility_spread",
        "classification",
        "combined_score",
    }

    missing = required_columns.difference(set(data.columns))

    if missing:
        raise ValueError(f"Candidates file missing columns: {missing}")

    return data.reset_index(drop=True)


def load_positions(
    file_path: str = POSITIONS_FILE,
) -> pd.DataFrame:
    """
    Load live paper positions CSV.

    If no positions file exists or file is empty, return empty DataFrame.
    """

    path = Path(file_path)

    if not path.exists():
        return pd.DataFrame()

    data = pd.read_csv(path)

    if data.empty:
        return data

    numeric_columns = [
        "spot_price",
        "strike",
        "days_to_expiry",
        "entry_price_usd",
        "quantity",
        "capital_at_risk",
        "model_price_usd",
        "price_diff_pct",
        "implied_volatility",
        "historical_volatility",
        "volatility_spread",
        "combined_score",
    ]

    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    return data.reset_index(drop=True)


def plot_market_vs_model_price(
    candidates: pd.DataFrame,
    show_plot: bool = True,
    save_path: str = "outputs/live_market_vs_model_price.png",
) -> None:
    """
    Plot Deribit market price vs Black-Scholes model price.
    """

    plot_data = candidates.dropna(
        subset=["market_price_usd", "model_price_usd", "classification"]
    ).copy()

    if plot_data.empty:
        print("No data for market vs model price plot.")
        return

    colors = (
        plot_data["classification"]
        .map(
            {
                "cheap": "green",
                "expensive": "red",
                "neutral": "gray",
            }
        )
        .fillna("gray")
        .astype(str)
        .tolist()
    )

    model_prices = pd.to_numeric(
        plot_data["model_price_usd"],
        errors="coerce",
    ).astype(float)

    market_prices = pd.to_numeric(
        plot_data["market_price_usd"],
        errors="coerce",
    ).astype(float)

    max_price = float(max(model_prices.max(), market_prices.max()))

    plt.figure(figsize=(10, 7))

    plt.scatter(
        x=model_prices.tolist(),
        y=market_prices.tolist(),
        c=colors,
        alpha=0.75,
    )

    plt.plot(
        [0.0, max_price],
        [0.0, max_price],
        linestyle="--",
        color="black",
        label="Market price = model price",
    )

    plt.title("ETH Options: Market Price vs Black-Scholes Model Price")
    plt.xlabel("Black-Scholes model price, USD")
    plt.ylabel("Deribit market price, USD")
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_iv_vs_historical_vol(
    candidates: pd.DataFrame,
    show_plot: bool = True,
    save_path: str = "outputs/live_iv_vs_historical_vol.png",
) -> None:
    """
    Plot implied volatility vs historical volatility.
    """

    plot_data = candidates.dropna(
        subset=["implied_volatility", "historical_volatility", "strike"]
    ).copy()

    if plot_data.empty:
        print("No data for IV vs historical volatility plot.")
        return

    strikes = pd.to_numeric(plot_data["strike"], errors="coerce").astype(float)
    implied_vols = pd.to_numeric(
        plot_data["implied_volatility"],
        errors="coerce",
    ).astype(float)

    historical_vol = float(plot_data["historical_volatility"].iloc[0])

    plt.figure(figsize=(12, 6))

    plt.scatter(
        x=strikes.tolist(),
        y=implied_vols.tolist(),
        label="Deribit mark IV",
        alpha=0.75,
    )

    plt.axhline(
        historical_vol,
        linestyle="--",
        color="black",
        label=f"Historical volatility: {historical_vol:.2%}",
    )

    plt.title("ETH Options: Implied Volatility vs Historical Volatility")
    plt.xlabel("Strike")
    plt.ylabel("Volatility")
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_volatility_spread_by_strike(
    candidates: pd.DataFrame,
    show_plot: bool = True,
    save_path: str = "outputs/live_volatility_spread_by_strike.png",
) -> None:
    """
    Plot volatility spread by strike.
    """

    plot_data = candidates.dropna(
        subset=["strike", "volatility_spread", "option_type"]
    ).copy()

    if plot_data.empty:
        print("No data for volatility spread plot.")
        return

    plt.figure(figsize=(12, 6))

    for option_type, group in plot_data.groupby("option_type"):
        strikes = pd.to_numeric(group["strike"], errors="coerce").astype(float)
        spreads = pd.to_numeric(
            group["volatility_spread"],
            errors="coerce",
        ).astype(float)

        plt.scatter(
            x=strikes.tolist(),
            y=spreads.tolist(),
            label=str(option_type).upper(),
            alpha=0.75,
        )

    plt.axhline(0.0, linestyle="--", color="black")

    plt.title("ETH Options: Volatility Spread by Strike")
    plt.xlabel("Strike")
    plt.ylabel("Implied volatility - historical volatility")
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_classification_counts(
    candidates: pd.DataFrame,
    show_plot: bool = True,
    save_path: str = "outputs/live_classification_counts.png",
) -> None:
    """
    Plot number of cheap, expensive, and neutral options.
    """

    if "classification" not in candidates.columns:
        print("No classification column. Skipping count plot.")
        return

    counts = candidates["classification"].astype(str).value_counts()

    if counts.empty:
        print("No classification data. Skipping count plot.")
        return

    labels = counts.index.astype(str).tolist()
    values = counts.astype(int).tolist()

    plt.figure(figsize=(8, 5))

    plt.bar(
        x=labels,
        height=values,
    )

    plt.title("ETH Options: Mispricing Classification Counts")
    plt.xlabel("Classification")
    plt.ylabel("Number of options")
    plt.grid(True, axis="y")

    plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_top_candidates(
    candidates: pd.DataFrame,
    top_n: int = 15,
    show_plot: bool = True,
    save_path: str = "outputs/live_top_candidates.png",
) -> None:
    """
    Plot top candidates by combined score.
    """

    plot_data = candidates.dropna(subset=["combined_score"]).copy()

    if plot_data.empty:
        print("No combined_score data. Skipping top candidates plot.")
        return

    plot_data = plot_data.sort_values(
        by="combined_score",
        ascending=False,
    ).head(top_n)

    labels = plot_data["instrument_name"].astype(str).tolist()

    scores = pd.to_numeric(
        plot_data["combined_score"],
        errors="coerce",
    ).fillna(0.0).astype(float).tolist()

    plt.figure(figsize=(12, 7))

    # Important:
    # Convert pandas Series to normal Python lists.
    # This fixes Pylance matplotlib type warnings.
    plt.barh(
        y=labels,
        width=scores,
    )

    plt.title(f"Top {top_n} ETH Option Candidates by Combined Score")
    plt.xlabel("Combined score")
    plt.ylabel("Instrument")
    plt.gca().invert_yaxis()
    plt.grid(True, axis="x")

    plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close()


def print_position_report(
    positions: pd.DataFrame,
) -> None:
    """
    Print paper positions summary.
    """

    print("\n========== LIVE PAPER POSITIONS REPORT ==========")

    if positions.empty:
        print("No paper positions were opened.")
        print("=================================================\n")
        return

    if "capital_at_risk" in positions.columns:
        total_risk = float(
            pd.to_numeric(
                positions["capital_at_risk"],
                errors="coerce",
            ).fillna(0.0).sum()
        )
    else:
        total_risk = 0.0

    print(f"Open paper positions: {len(positions)}")
    print(f"Total capital at risk: ${total_risk:,.2f}")

    display_columns = [
        "instrument_name",
        "option_type",
        "strike",
        "days_to_expiry",
        "entry_price_usd",
        "quantity",
        "capital_at_risk",
        "price_diff_pct",
        "volatility_spread",
        "combined_score",
    ]

    available_columns = [
        column for column in display_columns if column in positions.columns
    ]

    print()
    print(positions[available_columns].to_string(index=False))
    print("=================================================\n")


def generate_live_option_report(
    candidates_file: str = CANDIDATES_FILE,
    positions_file: str = POSITIONS_FILE,
    show_plot: bool = True,
) -> None:
    """
    Generate full live option visual report.
    """

    ensure_output_folder()

    candidates = load_candidates(candidates_file)
    positions = load_positions(positions_file)

    print("\n========== LIVE OPTION REPORT ==========")
    print(f"Candidates loaded: {len(candidates)}")
    print(f"Positions loaded:  {len(positions)}")
    print("========================================\n")

    plot_market_vs_model_price(candidates, show_plot=show_plot)
    plot_iv_vs_historical_vol(candidates, show_plot=show_plot)
    plot_volatility_spread_by_strike(candidates, show_plot=show_plot)
    plot_classification_counts(candidates, show_plot=show_plot)
    plot_top_candidates(candidates, top_n=15, show_plot=show_plot)

    print_position_report(positions)

    print("========== LIVE OPTION REPORT GENERATED ==========")
    print("Saved charts:")
    print("outputs/live_market_vs_model_price.png")
    print("outputs/live_iv_vs_historical_vol.png")
    print("outputs/live_volatility_spread_by_strike.png")
    print("outputs/live_classification_counts.png")
    print("outputs/live_top_candidates.png")
    print("==================================================")


if __name__ == "__main__":
    generate_live_option_report(
        candidates_file=CANDIDATES_FILE,
        positions_file=POSITIONS_FILE,
        show_plot=True,
    )