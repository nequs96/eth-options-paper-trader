"""
strategies/live_option_scanner.py

Live ETH options scanner using real Deribit market data.

This module:
- loads live ETH option chain from Deribit
- loads ETH historical prices
- estimates historical volatility
- prices options using Black-Scholes
- compares market price vs model price
- detects cheap / expensive options
- ranks opportunities
- saves scan results to CSV

Important:
This is for research and education only.
It does NOT place trades.
"""

from pathlib import Path

import pandas as pd

from data.market_data import download_eth_data, get_close_prices
from models.black_scholes import black_scholes_price
from models.volatility import historical_volatility
from strategies.option_mispricing import classify_option_mispricing


OUTPUT_FOLDER = "outputs"
OPTION_CHAIN_FILE = "outputs/live_eth_option_chain.csv"


def load_option_chain(file_path: str = OPTION_CHAIN_FILE) -> pd.DataFrame:
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
        raise ValueError("Option chain CSV is empty.")

    required_columns = {
        "instrument_name",
        "option_type",
        "strike",
        "days_to_expiry",
        "underlying_price",
        "mark_price_usd",
        "mark_iv",
    }

    missing = required_columns - set(data.columns)

    if missing:
        raise ValueError(f"Option chain missing columns: {missing}")

    data["strike"] = pd.to_numeric(data["strike"], errors="coerce")
    data["days_to_expiry"] = pd.to_numeric(data["days_to_expiry"], errors="coerce")
    data["underlying_price"] = pd.to_numeric(data["underlying_price"], errors="coerce")
    data["mark_price_usd"] = pd.to_numeric(data["mark_price_usd"], errors="coerce")
    data["mark_iv"] = pd.to_numeric(data["mark_iv"], errors="coerce")

    data = data.dropna(
        subset=[
            "strike",
            "days_to_expiry",
            "underlying_price",
            "mark_price_usd",
            "mark_iv",
        ]
    )

    return data.reset_index(drop=True)


def estimate_eth_historical_volatility(
    start_date: str = "2023-01-01",
) -> float:
    """
    Estimate ETH historical volatility using daily data.
    """

    data = download_eth_data(
        start_date=start_date,
        end_date=None,
        interval="1d",
    )

    close_prices = get_close_prices(data)

    vol = historical_volatility(
        prices=close_prices,
        timeframe="1d",
        use_log_returns=True,
    )

    return float(vol)


def price_option_black_scholes(
    spot_price: float,
    strike_price: float,
    days_to_expiry: float,
    volatility: float,
    option_type: str,
    risk_free_rate: float = 0.04,
) -> float:
    """
    Black-Scholes option pricing wrapper.
    """

    time_to_expiry = max(days_to_expiry, 0.0) / 365.0

    return black_scholes_price(
        S=spot_price,
        K=strike_price,
        T=time_to_expiry,
        r=risk_free_rate,
        sigma=volatility,
        option_type=option_type,
    )


def scan_live_options() -> pd.DataFrame:
    """
    Scan live ETH options for mispricing.
    """

    print("Loading live ETH option chain...")
    option_chain = load_option_chain()

    print("Estimating ETH historical volatility...")
    hist_vol = estimate_eth_historical_volatility()

    print(f"Estimated ETH historical volatility: {hist_vol:.2%}")

    scan_rows = []

    for _, row in option_chain.iterrows():
        spot = float(row["underlying_price"])
        strike = float(row["strike"])
        dte = float(row["days_to_expiry"])
        option_type = str(row["option_type"])
        market_price = float(row["mark_price_usd"])
        implied_vol = float(row["mark_iv"])

        if spot <= 0 or strike <= 0 or dte <= 0:
            continue

        try:
            model_price = price_option_black_scholes(
                spot_price=spot,
                strike_price=strike,
                days_to_expiry=dte,
                volatility=hist_vol,
                option_type=option_type,
            )
        except Exception:
            continue

        price_diff = market_price - model_price
        price_diff_pct = price_diff / model_price if model_price > 0 else 0.0
        vol_spread = implied_vol - hist_vol

        classification = classify_option_mispricing(
            price_difference_percent=price_diff_pct,
            vol_spread=vol_spread,
            price_threshold=0.10,
            volatility_threshold=0.10,
        )

        scan_rows.append(
            {
                "instrument_name": row["instrument_name"],
                "option_type": option_type,
                "strike": strike,
                "days_to_expiry": dte,
                "spot_price": spot,
                "market_price_usd": market_price,
                "model_price_usd": model_price,
                "price_diff_usd": price_diff,
                "price_diff_pct": price_diff_pct,
                "implied_volatility": implied_vol,
                "historical_volatility": hist_vol,
                "volatility_spread": vol_spread,
                "classification": classification,
            }
        )

    scan_df = pd.DataFrame(scan_rows)

    if scan_df.empty:
        raise RuntimeError("No options were successfully scanned.")

    scan_df = scan_df.sort_values(
        by="price_diff_pct",
        ascending=True,
    ).reset_index(drop=True)

    output_path = Path(OUTPUT_FOLDER) / "live_option_scan.csv"
    scan_df.to_csv(output_path, index=False)

    print(f"Saved live option scan to: {output_path}")

    return scan_df


def print_scan_summary(scan_df: pd.DataFrame, top_n: int = 15) -> None:
    """
    Print top mispriced options.
    """

    print("\n========== LIVE OPTION SCAN SUMMARY ==========")

    cheap = scan_df[scan_df["classification"] == "cheap"].head(top_n)
    expensive = scan_df[scan_df["classification"] == "expensive"].head(top_n)

    if not cheap.empty:
        print("\n--- CHEAP OPTIONS ---")
        print(
            cheap[
                [
                    "instrument_name",
                    "option_type",
                    "strike",
                    "days_to_expiry",
                    "market_price_usd",
                    "model_price_usd",
                    "price_diff_pct",
                    "volatility_spread",
                ]
            ].to_string(index=False)
        )

    if not expensive.empty:
        print("\n--- EXPENSIVE OPTIONS ---")
        print(
            expensive[
                [
                    "instrument_name",
                    "option_type",
                    "strike",
                    "days_to_expiry",
                    "market_price_usd",
                    "model_price_usd",
                    "price_diff_pct",
                    "volatility_spread",
                ]
            ].to_string(index=False)
        )

    print("\n==============================================")


if __name__ == "__main__":
    scan = scan_live_options()
    print_scan_summary(scan)