"""
strategies/live_option_scanner.py

Live ETH options scanner using real Deribit market data.

This module:
- loads live ETH option chain from Deribit CSV
- loads ETH historical prices
- estimates historical volatility
- analyzes option mispricing
- applies risk rules
- ranks opportunities
- saves scan results to CSV

Important:
This is for research and education only.
It does NOT place trades.
"""

from pathlib import Path
import math

import pandas as pd

from data.market_data import download_eth_data, get_close_prices
from models.volatility import historical_volatility
from strategies.option_mispricing import analyze_option_mispricing
from strategies.risk_rules import approve_trade


OUTPUT_FOLDER = "outputs"
OPTION_CHAIN_FILE = "outputs/live_eth_option_chain.csv"

RISK_FREE_RATE = 0.04

# Scanner thresholds
MIN_DAYS_TO_EXPIRY = 1
MAX_DAYS_TO_EXPIRY = 60
MIN_MARK_PRICE_USD = 0.01

# Mispricing thresholds
PRICE_THRESHOLD = 0.10
VOLATILITY_THRESHOLD = 0.10

# Risk assumptions for long option candidates
DEFAULT_CAPITAL = 10_000.0
MAX_RISK_PER_TRADE = 0.01
MAX_POSITION_PCT = 0.10
MAX_DAILY_DRAWDOWN = 0.02

# Long option exit assumptions.
# These are research assumptions, not execution orders.
STOP_LOSS_PCT = 0.50       # exit if option loses 50%
TAKE_PROFIT_PCT = 1.00     # exit if option gains 100%

# Crypto options usually use multiplier 1, but keep configurable.
CONTRACT_MULTIPLIER = 1.0


def _is_valid_number(value: float) -> bool:
    """
    Return True if value is a finite int/float.
    """
    return isinstance(value, (int, float)) and math.isfinite(value)


def normalize_deribit_iv(mark_iv: float) -> float:
    """
    Normalize Deribit mark_iv.

    Some feeds store IV as:
        85.0 meaning 85%

    Other feeds store IV as:
        0.85 meaning 85%

    This function converts both into decimal format.
    """

    if not _is_valid_number(mark_iv) or mark_iv <= 0:
        return 0.0

    # If mark_iv is above 3, assume it is percent form.
    # Example: 85 -> 0.85
    if mark_iv > 3:
        return mark_iv / 100.0

    return float(mark_iv)


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

    numeric_columns = [
        "strike",
        "days_to_expiry",
        "underlying_price",
        "mark_price_usd",
        "mark_iv",
    ]

    for col in numeric_columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=numeric_columns)

    data["option_type"] = data["option_type"].astype(str).str.lower().str.strip()

    data = data[data["option_type"].isin(["call", "put"])]

    data = data[
        (data["strike"] > 0)
        & (data["days_to_expiry"] > 0)
        & (data["underlying_price"] > 0)
        & (data["mark_price_usd"] > 0)
        & (data["mark_iv"] > 0)
    ]

    data = data.reset_index(drop=True)

    if data.empty:
        raise ValueError("No valid option rows after cleaning.")

    return data


def estimate_eth_historical_volatility(
    start_date: str = "2023-01-01",
) -> float:
    """
    Estimate ETH historical volatility using daily historical data.
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

    if not _is_valid_number(vol) or vol <= 0:
        raise ValueError("Failed to estimate valid ETH historical volatility.")

    return float(vol)


def build_long_option_exit_plan(
    market_price: float,
    stop_loss_pct: float = STOP_LOSS_PCT,
    take_profit_pct: float = TAKE_PROFIT_PCT,
) -> tuple[float, float, float]:
    """
    Build simple research exit plan for a long option.

    entry_price:
        current market price

    stop_loss:
        price where option is assumed to be closed for loss

    take_profit:
        price where option is assumed to be closed for profit
    """

    entry_price = float(market_price)
    stop_loss = entry_price * (1.0 - stop_loss_pct)
    take_profit = entry_price * (1.0 + take_profit_pct)

    return entry_price, stop_loss, take_profit


def should_mark_as_trade_candidate(
    classification: str,
    confidence_level: str,
    risk_allowed: bool,
    mispricing_score: float,
) -> bool:
    """
    Decide whether scanner should mark an option as a candidate.

    For now, only cheap options are marked as long-option candidates.

    Expensive options are research signals only because trading expensive options
    usually means selling options, and short-option risk requires a different
    max-loss model.
    """

    if classification != "cheap":
        return False

    if confidence_level not in {"medium", "high"}:
        return False

    if mispricing_score > -0.05:
        return False

    if not risk_allowed:
        return False

    return True


def scan_live_options(
    capital: float = DEFAULT_CAPITAL,
    starting_day_equity: float | None = None,
    current_equity: float | None = None,
) -> pd.DataFrame:
    """
    Scan live ETH options for mispricing and risk-approved candidates.

    Parameters
    ----------
    capital : float
        Account capital used for risk sizing.
    starting_day_equity : float | None
        Optional equity at start of day.
    current_equity : float | None
        Optional current equity.

    Returns
    -------
    pd.DataFrame
        Full scan results.
    """

    print("Loading live ETH option chain...")
    option_chain = load_option_chain()

    print("Estimating ETH historical volatility...")
    hist_vol = estimate_eth_historical_volatility()

    print(f"Estimated ETH historical volatility: {hist_vol:.2%}")

    if starting_day_equity is None:
        starting_day_equity = capital

    if current_equity is None:
        current_equity = capital

    scan_rows = []

    for _, row in option_chain.iterrows():
        instrument_name = str(row["instrument_name"])
        option_type = str(row["option_type"]).lower().strip()

        spot = float(row["underlying_price"])
        strike = float(row["strike"])
        dte = float(row["days_to_expiry"])
        market_price = float(row["mark_price_usd"])
        implied_vol_from_feed = normalize_deribit_iv(float(row["mark_iv"]))

        if dte < MIN_DAYS_TO_EXPIRY or dte > MAX_DAYS_TO_EXPIRY:
            continue

        if market_price < MIN_MARK_PRICE_USD:
            continue

        time_to_expiry = dte / 365.0

        # =========================
        # Mispricing analysis
        # =========================

        result = analyze_option_mispricing(
            market_price=market_price,
            spot_price=spot,
            strike_price=strike,
            time_to_expiry=time_to_expiry,
            risk_free_rate=RISK_FREE_RATE,
            historical_volatility=hist_vol,
            option_type=option_type,
            price_threshold=PRICE_THRESHOLD,
            volatility_threshold=VOLATILITY_THRESHOLD,
        )

        # If mispricing module rejects data, still record row for debugging.
        if result.classification == "invalid":
            scan_rows.append(
                {
                    "instrument_name": instrument_name,
                    "option_type": option_type,
                    "strike": strike,
                    "days_to_expiry": dte,
                    "spot_price": spot,
                    "market_price_usd": market_price,
                    "model_price_usd": 0.0,
                    "price_diff_usd": 0.0,
                    "price_diff_pct": 0.0,
                    "implied_volatility_model": 0.0,
                    "implied_volatility_feed": implied_vol_from_feed,
                    "historical_volatility": hist_vol,
                    "volatility_spread": 0.0,
                    "classification": "invalid",
                    "confidence_level": "none",
                    "mispricing_score": 0.0,
                    "risk_allowed": False,
                    "risk_position_size": 0.0,
                    "risk_amount": 0.0,
                    "risk_reward": 0.0,
                    "trade_candidate": False,
                    "rejection_reason": result.explanation,
                }
            )
            continue

        # =========================
        # Risk rules
        # =========================

        entry_price, stop_loss, take_profit = build_long_option_exit_plan(
            market_price=market_price,
        )

        risk_decision = approve_trade(
            capital=capital,
            option_price=market_price,
            implied_volatility=result.implied_volatility,
            historical_volatility=hist_vol,
            max_risk_per_trade=MAX_RISK_PER_TRADE,
            starting_day_equity=starting_day_equity,
            current_equity=current_equity,
            max_daily_drawdown=MAX_DAILY_DRAWDOWN,
            max_position_pct=MAX_POSITION_PCT,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            contract_multiplier=CONTRACT_MULTIPLIER,
            allow_fractional_size=True,
        )

        trade_candidate = should_mark_as_trade_candidate(
            classification=result.classification,
            confidence_level=result.confidence_level,
            risk_allowed=risk_decision.allowed,
            mispricing_score=result.mispricing_score,
        )

        rejection_reason = ""

        if not trade_candidate:
            if result.classification != "cheap":
                rejection_reason = "Not a cheap long-option candidate."
            elif result.confidence_level not in {"medium", "high"}:
                rejection_reason = f"Confidence too low: {result.confidence_level}."
            elif not risk_decision.allowed:
                rejection_reason = risk_decision.reason
            else:
                rejection_reason = "Candidate filters not met."

        scan_rows.append(
            {
                "instrument_name": instrument_name,
                "option_type": option_type,
                "strike": strike,
                "days_to_expiry": dte,
                "spot_price": spot,
                "market_price_usd": market_price,
                "model_price_usd": result.theoretical_price,
                "price_diff_usd": result.price_difference,
                "price_diff_pct": result.price_difference_percent,
                "implied_volatility_model": result.implied_volatility,
                "implied_volatility_feed": implied_vol_from_feed,
                "historical_volatility": hist_vol,
                "volatility_spread": result.volatility_spread,
                "volatility_spread_pp": result.volatility_spread_percent_points,
                "classification": result.classification,
                "research_signal": result.signal,
                "confidence_level": result.confidence_level,
                "mispricing_score": result.mispricing_score,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "risk_allowed": risk_decision.allowed,
                "risk_position_size": risk_decision.position_size,
                "risk_amount": risk_decision.risk_amount,
                "max_loss_allowed": risk_decision.max_loss_allowed,
                "position_notional": risk_decision.position_notional,
                "risk_reward": risk_decision.risk_reward,
                "daily_drawdown": risk_decision.daily_drawdown,
                "trade_candidate": trade_candidate,
                "rejection_reason": rejection_reason,
                "explanation": result.explanation,
            }
        )

    scan_df = pd.DataFrame(scan_rows)

    if scan_df.empty:
        raise RuntimeError("No options were successfully scanned.")

    # Rank candidates first, then strongest cheap opportunities.
    scan_df = scan_df.sort_values(
        by=[
            "trade_candidate",
            "classification",
            "confidence_level",
            "mispricing_score",
            "price_diff_pct",
        ],
        ascending=[False, True, True, True, True],
    ).reset_index(drop=True)

    output_folder = Path(OUTPUT_FOLDER)
    output_folder.mkdir(parents=True, exist_ok=True)

    output_path = output_folder / "live_option_scan.csv"
    scan_df.to_csv(output_path, index=False)

    candidate_path = output_folder / "live_option_candidates.csv"
    scan_df[scan_df["trade_candidate"] == True].to_csv(candidate_path, index=False)

    print(f"Saved full live option scan to: {output_path}")
    print(f"Saved trade candidates to: {candidate_path}")

    return scan_df


def print_scan_summary(scan_df: pd.DataFrame, top_n: int = 15) -> None:
    """
    Print top mispriced options and risk-approved candidates.
    """

    print("\n========== LIVE OPTION SCAN SUMMARY ==========")

    candidates = scan_df[scan_df["trade_candidate"] == True].head(top_n)
    cheap = scan_df[scan_df["classification"] == "cheap"].head(top_n)
    expensive = scan_df[scan_df["classification"] == "expensive"].head(top_n)

    if not candidates.empty:
        print("\n--- RISK-APPROVED LONG OPTION CANDIDATES ---")
        print(
            candidates[
                [
                    "instrument_name",
                    "option_type",
                    "strike",
                    "days_to_expiry",
                    "market_price_usd",
                    "model_price_usd",
                    "price_diff_pct",
                    "confidence_level",
                    "mispricing_score",
                    "risk_position_size",
                    "risk_amount",
                    "risk_reward",
                ]
            ].to_string(index=False)
        )
    else:
        print("\nNo risk-approved long option candidates found.")

    if not cheap.empty:
        print("\n--- CHEAP OPTIONS RESEARCH SIGNALS ---")
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
                    "confidence_level",
                    "mispricing_score",
                    "risk_allowed",
                    "rejection_reason",
                ]
            ].to_string(index=False)
        )

    if not expensive.empty:
        print("\n--- EXPENSIVE OPTIONS RESEARCH SIGNALS ---")
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
                    "confidence_level",
                    "mispricing_score",
                    "rejection_reason",
                ]
            ].to_string(index=False)
        )

    print("\n==============================================")


if __name__ == "__main__":
    scan = scan_live_options(
        capital=10_000.0,
        starting_day_equity=10_000.0,
        current_equity=10_000.0,
    )

    print_scan_summary(scan)