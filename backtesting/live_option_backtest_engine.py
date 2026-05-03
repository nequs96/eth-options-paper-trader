"""
backtesting/live_option_backtest_engine.py

Live ETH options paper-backtest engine using real Deribit option-chain data.

This module:
- fetches or loads live ETH option chain from Deribit
- estimates ETH historical volatility
- calculates Black-Scholes model prices
- compares real market prices vs model prices
- classifies options as cheap / expensive / neutral
- applies risk rules
- opens simulated paper positions
- saves candidate list and paper positions

Important:
This is NOT a historical options backtest.
It is a live snapshot paper simulation.

A true historical options backtest requires historical option-chain snapshots,
historical bid/ask prices, historical IV, and historical Greeks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from data.market_data import download_eth_data, get_close_prices
from data.options_data import (
    DeribitConfig,
    build_live_eth_option_chain,
)
from models.black_scholes import black_scholes_price
from models.volatility import historical_volatility
from strategies.option_mispricing import classify_option_mispricing
from strategies.risk_rules import approve_trade
from backtesting.portfolio import Portfolio, print_portfolio_summary


@dataclass
class LiveOptionBacktestConfig:
    """
    Configuration for live option paper-backtest.
    """

    refresh_option_chain: bool = True
    option_chain_file: str = "outputs/live_eth_option_chain.csv"

    output_folder: str = "outputs"

    historical_vol_start_date: str = "2023-01-01"

    initial_cash: float = 10_000.0
    max_risk_per_trade: float = 0.01
    max_positions: int = 5

    risk_free_rate: float = 0.04

    min_days_to_expiry: float = 3.0
    max_days_to_expiry: float = 45.0

    min_market_price_usd: float = 5.0
    max_bid_ask_spread_pct: float = 0.35

    price_threshold: float = 0.10
    volatility_threshold: float = 0.10

    min_volatility: float = 0.10
    max_volatility: float = 2.50

    allow_calls: bool = True
    allow_puts: bool = True

    only_trade_cheap_options: bool = True


@dataclass
class LiveOptionBacktestResult:
    """
    Output container.
    """

    candidates: pd.DataFrame
    paper_positions: pd.DataFrame
    final_cash: float
    open_positions: int


def ensure_output_folder(folder: str) -> None:
    """
    Create output folder if needed.
    """

    Path(folder).mkdir(parents=True, exist_ok=True)


def load_live_option_chain(file_path: str) -> pd.DataFrame:
    """
    Load live option chain from CSV.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Option chain file not found: {file_path}. "
            "Run: python -m data.options_data"
        )

    data = pd.read_csv(path)

    if data.empty:
        raise ValueError("Live option chain CSV is empty.")

    required_columns = {
        "instrument_name",
        "option_type",
        "strike",
        "days_to_expiry",
        "underlying_price",
        "mark_price_usd",
        "mark_iv",
    }

    missing = required_columns.difference(set(data.columns))

    if missing:
        raise ValueError(f"Live option chain missing columns: {missing}")

    numeric_columns = [
        "strike",
        "days_to_expiry",
        "underlying_price",
        "best_bid_price_usd",
        "best_ask_price_usd",
        "mark_price_usd",
        "last_price_usd",
        "mark_iv",
        "bid_iv",
        "ask_iv",
        "delta",
        "gamma",
        "theta",
        "vega",
        "rho",
        "open_interest",
    ]

    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(
        subset=[
            "instrument_name",
            "option_type",
            "strike",
            "days_to_expiry",
            "underlying_price",
            "mark_price_usd",
            "mark_iv",
        ]
    ).copy()

    return data.reset_index(drop=True)


def get_or_refresh_live_option_chain(
    config: LiveOptionBacktestConfig,
) -> pd.DataFrame:
    """
    Refresh option chain from Deribit or load existing CSV.
    """

    if config.refresh_option_chain:
        print("Refreshing live Deribit ETH option chain...")

        deribit_config = DeribitConfig(
            currency="ETH",
            kind="option",
            testnet=False,
            request_sleep_seconds=0.15,
            output_folder=config.output_folder,
        )

        option_chain = build_live_eth_option_chain(
            config=deribit_config,
            min_days_to_expiry=int(config.min_days_to_expiry),
            max_days_to_expiry=int(config.max_days_to_expiry),
            strikes_each_side=5,
            save_csv=True,
        )

        return option_chain

    print("Loading existing live ETH option chain CSV...")

    return load_live_option_chain(config.option_chain_file)


def estimate_current_historical_volatility(
    start_date: str,
) -> float:
    """
    Estimate ETH historical volatility from daily ETH data.
    """

    print("Downloading ETH historical data for volatility estimate...")

    data = download_eth_data(
        start_date=start_date,
        end_date=None,
        interval="1d",
    )

    close_prices = get_close_prices(data)

    volatility = historical_volatility(
        prices=close_prices,
        timeframe="1d",
        use_log_returns=True,
    )

    return float(volatility)


def choose_market_price_usd(row: pd.Series) -> float | None:
    """
    Choose a usable market option price in USD.

    Preference:
    1. mid price from bid/ask if both exist and positive
    2. mark price
    3. last price
    """

    bid = row.get("best_bid_price_usd")
    ask = row.get("best_ask_price_usd")
    mark = row.get("mark_price_usd")
    last = row.get("last_price_usd")

    bid_float = safe_optional_float(bid)
    ask_float = safe_optional_float(ask)
    mark_float = safe_optional_float(mark)
    last_float = safe_optional_float(last)

    if (
        bid_float is not None
        and ask_float is not None
        and bid_float > 0
        and ask_float > 0
        and ask_float >= bid_float
    ):
        return float((bid_float + ask_float) / 2.0)

    if mark_float is not None and mark_float > 0:
        return float(mark_float)

    if last_float is not None and last_float > 0:
        return float(last_float)

    return None


def calculate_bid_ask_spread_pct(row: pd.Series) -> float | None:
    """
    Calculate bid/ask spread percentage.

    Formula:
        spread_pct = (ask - bid) / mid
    """

    bid = safe_optional_float(row.get("best_bid_price_usd"))
    ask = safe_optional_float(row.get("best_ask_price_usd"))

    if bid is None or ask is None:
        return None

    if bid <= 0 or ask <= 0 or ask < bid:
        return None

    mid = (bid + ask) / 2.0

    if mid <= 0:
        return None

    return float((ask - bid) / mid)


def safe_optional_float(value: Any) -> float | None:
    """
    Convert value to float or None.
    """

    if value is None:
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if pd.isna(number):
        return None

    return float(number)


def calculate_model_price(
    spot_price: float,
    strike_price: float,
    days_to_expiry: float,
    historical_volatility_value: float,
    risk_free_rate: float,
    option_type: str,
) -> float:
    """
    Calculate Black-Scholes model price in USD.
    """

    time_to_expiry = max(days_to_expiry, 0.0) / 365.0

    return black_scholes_price(
        S=spot_price,
        K=strike_price,
        T=time_to_expiry,
        r=risk_free_rate,
        sigma=historical_volatility_value,
        option_type=option_type,
    )


def filter_option_chain(
    option_chain: pd.DataFrame,
    config: LiveOptionBacktestConfig,
) -> pd.DataFrame:
    """
    Apply basic filters to live option chain.
    """

    data = option_chain.copy()

    data = data[
        (data["days_to_expiry"] >= config.min_days_to_expiry)
        & (data["days_to_expiry"] <= config.max_days_to_expiry)
    ].copy()

    if not config.allow_calls:
        data = data[data["option_type"] != "call"].copy()

    if not config.allow_puts:
        data = data[data["option_type"] != "put"].copy()

    if data.empty:
        return data

    data["market_price_usd"] = data.apply(choose_market_price_usd, axis=1)
    data["bid_ask_spread_pct"] = data.apply(calculate_bid_ask_spread_pct, axis=1)

    data = data.dropna(subset=["market_price_usd"]).copy()

    data = data[data["market_price_usd"] >= config.min_market_price_usd].copy()

    # If spread is missing, keep the option but mark spread as unknown.
    data = data[
        data["bid_ask_spread_pct"].isna()
        | (data["bid_ask_spread_pct"] <= config.max_bid_ask_spread_pct)
    ].copy()

    return data.reset_index(drop=True)


def build_candidate_table(
    option_chain: pd.DataFrame,
    historical_volatility_value: float,
    config: LiveOptionBacktestConfig,
) -> pd.DataFrame:
    """
    Build candidate table with model-vs-market comparison.
    """

    rows: list[dict[str, Any]] = []

    for _, row in option_chain.iterrows():
        instrument_name = str(row["instrument_name"])
        option_type = str(row["option_type"]).lower().strip()

        spot_price = float(row["underlying_price"])
        strike_price = float(row["strike"])
        days_to_expiry = float(row["days_to_expiry"])
        market_price = float(row["market_price_usd"])

        implied_volatility = safe_optional_float(row.get("mark_iv"))

        if implied_volatility is None or implied_volatility <= 0:
            continue

        if option_type not in {"call", "put"}:
            continue

        if spot_price <= 0 or strike_price <= 0 or days_to_expiry <= 0:
            continue

        if market_price <= 0:
            continue

        try:
            model_price = calculate_model_price(
                spot_price=spot_price,
                strike_price=strike_price,
                days_to_expiry=days_to_expiry,
                historical_volatility_value=historical_volatility_value,
                risk_free_rate=config.risk_free_rate,
                option_type=option_type,
            )
        except Exception:
            continue

        if model_price <= 0:
            continue

        price_diff = market_price - model_price
        price_diff_pct = price_diff / model_price

        volatility_spread = implied_volatility - historical_volatility_value

        classification = classify_option_mispricing(
            price_difference_percent=price_diff_pct,
            vol_spread=volatility_spread,
            price_threshold=config.price_threshold,
            volatility_threshold=config.volatility_threshold,
        )

        # Cheap score:
        # positive means model thinks option is undervalued.
        cheapness_score = (model_price - market_price) / model_price

        # Vol edge:
        # positive means Deribit IV is below historical vol.
        volatility_edge = historical_volatility_value - implied_volatility

        combined_score = cheapness_score + volatility_edge

        rows.append(
            {
                "instrument_name": instrument_name,
                "option_type": option_type,
                "spot_price": spot_price,
                "strike": strike_price,
                "days_to_expiry": days_to_expiry,
                "market_price_usd": market_price,
                "model_price_usd": model_price,
                "price_diff_usd": price_diff,
                "price_diff_pct": price_diff_pct,
                "implied_volatility": implied_volatility,
                "historical_volatility": historical_volatility_value,
                "volatility_spread": volatility_spread,
                "cheapness_score": cheapness_score,
                "volatility_edge": volatility_edge,
                "combined_score": combined_score,
                "bid_ask_spread_pct": row.get("bid_ask_spread_pct"),
                "delta": row.get("delta"),
                "gamma": row.get("gamma"),
                "theta": row.get("theta"),
                "vega": row.get("vega"),
                "open_interest": row.get("open_interest"),
                "classification": classification,
            }
        )

    candidates = pd.DataFrame(rows)

    if candidates.empty:
        return candidates

    candidates = candidates.sort_values(
        by=["classification", "combined_score"],
        ascending=[True, False],
    ).reset_index(drop=True)

    return candidates


def create_paper_positions(
    candidates: pd.DataFrame,
    config: LiveOptionBacktestConfig,
) -> tuple[Portfolio, pd.DataFrame]:
    """
    Create simulated paper positions from candidate table.
    """

    portfolio = Portfolio(initial_cash=config.initial_cash)

    position_rows: list[dict[str, Any]] = []

    if candidates.empty:
        return portfolio, pd.DataFrame(position_rows)

    trade_candidates = candidates.copy()

    if config.only_trade_cheap_options:
        trade_candidates = trade_candidates[
            trade_candidates["classification"] == "cheap"
        ].copy()

    if trade_candidates.empty:
        return portfolio, pd.DataFrame(position_rows)

    trade_candidates = trade_candidates.sort_values(
        by="combined_score",
        ascending=False,
    ).reset_index(drop=True)

    for _, row in trade_candidates.iterrows():
        if portfolio.number_of_open_positions() >= config.max_positions:
            break

        market_price = float(row["market_price_usd"])
        implied_volatility = float(row["implied_volatility"])
        historical_volatility_value = float(row["historical_volatility"])

        risk_decision = approve_trade(
            capital=portfolio.cash,
            option_price=market_price,
            implied_volatility=implied_volatility,
            historical_volatility=historical_volatility_value,
            max_risk_per_trade=config.max_risk_per_trade,
            max_volatility=config.max_volatility,
            min_volatility=config.min_volatility,
        )

        if not risk_decision.allowed:
            continue

        if risk_decision.position_size <= 0:
            continue

        try:
            portfolio.open_position(
                option_type=str(row["option_type"]),
                entry_price=market_price,
                quantity=risk_decision.position_size,
                strike_price=float(row["strike"]),
                days_to_expiry=max(int(float(row["days_to_expiry"])), 1),
                direction="long",
            )
        except ValueError:
            continue

        position_rows.append(
            {
                "instrument_name": row["instrument_name"],
                "option_type": row["option_type"],
                "spot_price": row["spot_price"],
                "strike": row["strike"],
                "days_to_expiry": row["days_to_expiry"],
                "entry_price_usd": market_price,
                "quantity": risk_decision.position_size,
                "capital_at_risk": risk_decision.risk_amount,
                "model_price_usd": row["model_price_usd"],
                "price_diff_pct": row["price_diff_pct"],
                "implied_volatility": row["implied_volatility"],
                "historical_volatility": row["historical_volatility"],
                "volatility_spread": row["volatility_spread"],
                "combined_score": row["combined_score"],
                "classification": row["classification"],
                "risk_reason": risk_decision.reason,
            }
        )

    positions = pd.DataFrame(position_rows)

    return portfolio, positions


def print_live_backtest_summary(
    candidates: pd.DataFrame,
    positions: pd.DataFrame,
    portfolio: Portfolio,
) -> None:
    """
    Print readable summary.
    """

    print("\n========== LIVE OPTION PAPER-BACKTEST SUMMARY ==========")

    print(f"Scanned candidates:       {len(candidates)}")

    if not candidates.empty:
        print(
            "Cheap options:            "
            f"{len(candidates[candidates['classification'] == 'cheap'])}"
        )
        print(
            "Expensive options:        "
            f"{len(candidates[candidates['classification'] == 'expensive'])}"
        )
        print(
            "Neutral options:          "
            f"{len(candidates[candidates['classification'] == 'neutral'])}"
        )

    print(f"Paper positions opened:   {len(positions)}")
    print(f"Remaining cash:           ${portfolio.cash:,.2f}")
    print("========================================================\n")

    if not positions.empty:
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

        print("--- PAPER POSITIONS ---")
        print(positions[display_columns].to_string(index=False))
        print()

    print_portfolio_summary(portfolio)


def run_live_option_paper_backtest(
    config: LiveOptionBacktestConfig | None = None,
) -> LiveOptionBacktestResult:
    """
    Run live option paper-backtest.
    """

    if config is None:
        config = LiveOptionBacktestConfig()

    ensure_output_folder(config.output_folder)

    option_chain = get_or_refresh_live_option_chain(config)

    filtered_chain = filter_option_chain(
        option_chain=option_chain,
        config=config,
    )

    if filtered_chain.empty:
        raise RuntimeError("No options passed live-chain filters.")

    historical_volatility_value = estimate_current_historical_volatility(
        start_date=config.historical_vol_start_date,
    )

    print(f"Estimated ETH historical volatility: {historical_volatility_value:.2%}")

    candidates = build_candidate_table(
        option_chain=filtered_chain,
        historical_volatility_value=historical_volatility_value,
        config=config,
    )

    if candidates.empty:
        raise RuntimeError("No valid option candidates were generated.")

    portfolio, positions = create_paper_positions(
        candidates=candidates,
        config=config,
    )

    candidates_output = Path(config.output_folder) / "live_backtest_candidates.csv"
    positions_output = Path(config.output_folder) / "live_paper_positions.csv"

    candidates.to_csv(candidates_output, index=False)
    positions.to_csv(positions_output, index=False)

    print(f"Saved candidates to:      {candidates_output}")
    print(f"Saved paper positions to: {positions_output}")

    print_live_backtest_summary(
        candidates=candidates,
        positions=positions,
        portfolio=portfolio,
    )

    return LiveOptionBacktestResult(
        candidates=candidates,
        paper_positions=positions,
        final_cash=float(portfolio.cash),
        open_positions=portfolio.number_of_open_positions(),
    )


if __name__ == "__main__":
    live_config = LiveOptionBacktestConfig(
        refresh_option_chain=True,
        option_chain_file="outputs/live_eth_option_chain.csv",
        output_folder="outputs",
        historical_vol_start_date="2023-01-01",
        initial_cash=10_000.0,
        max_risk_per_trade=0.01,
        max_positions=5,
        risk_free_rate=0.04,
        min_days_to_expiry=3.0,
        max_days_to_expiry=45.0,
        min_market_price_usd=5.0,
        max_bid_ask_spread_pct=0.35,
        price_threshold=0.10,
        volatility_threshold=0.10,
        min_volatility=0.10,
        max_volatility=2.50,
        allow_calls=True,
        allow_puts=True,
        only_trade_cheap_options=True,
    )

    run_live_option_paper_backtest(live_config)