"""
execution/paper_trader.py

Persistent paper trading module for ETH options.

This module:
- reads candidate options from live paper-backtest output
- opens simulated paper trades
- stores paper positions in CSV
- stores trade history in CSV
- does NOT place real orders

Important:
This is for research and education only.
It does not connect to an exchange account.
It does not execute real trades.
It is not financial advice.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class PaperTraderConfig:
    """
    Configuration for paper trader.
    """

    candidates_file: str = "outputs/live_backtest_candidates.csv"
    positions_file: str = "outputs/paper_open_positions.csv"
    trade_history_file: str = "outputs/paper_trade_history.csv"

    initial_cash_file: str = "outputs/paper_cash.csv"
    initial_cash: float = 10_000.0

    max_positions: int = 5
    max_risk_per_trade: float = 0.01

    only_trade_cheap_options: bool = True

    # Minimum absolute signal strength.
    # Works with either:
    # - combined_score
    # - abs(mispricing_score)
    min_abs_mispricing_score: float = 0.05

    min_market_price_usd: float = 5.0


def ensure_parent_folder(file_path: str) -> None:
    """
    Ensure parent folder exists.
    """

    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def safe_float(value: Any) -> float | None:
    """
    Convert value to float safely.
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


def load_csv_if_exists(file_path: str) -> pd.DataFrame:
    """
    Load CSV safely.

    Returns empty DataFrame if file does not exist or is empty.
    """

    path = Path(file_path)

    if not path.exists():
        return pd.DataFrame()

    if path.stat().st_size == 0:
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def load_cash(config: PaperTraderConfig) -> float:
    """
    Load paper cash balance.

    If cash file does not exist, create it with initial cash.
    """

    cash_path = Path(config.initial_cash_file)

    if not cash_path.exists():
        ensure_parent_folder(config.initial_cash_file)

        cash_data = pd.DataFrame(
            [
                {
                    "cash": float(config.initial_cash),
                }
            ]
        )

        cash_data.to_csv(cash_path, index=False)

        return float(config.initial_cash)

    data = pd.read_csv(cash_path)

    if data.empty or "cash" not in data.columns:
        raise ValueError("Paper cash file is invalid.")

    cash = safe_float(data["cash"].iloc[0])

    if cash is None:
        raise ValueError("Paper cash value is invalid.")

    return float(cash)


def save_cash(
    cash: float,
    config: PaperTraderConfig,
) -> None:
    """
    Save paper cash balance.
    """

    ensure_parent_folder(config.initial_cash_file)

    data = pd.DataFrame(
        [
            {
                "cash": float(cash),
            }
        ]
    )

    data.to_csv(config.initial_cash_file, index=False)


def load_candidates(config: PaperTraderConfig) -> pd.DataFrame:
    """
    Load candidate options from live paper-backtest/scanner output.

    This function is intentionally compatible with multiple candidate formats.

    Supported score columns:
    - combined_score
    - mispricing_score
    - cheapness_score + volatility_edge

    If missing:
    - combined_score is created from abs(mispricing_score)
    - mispricing_score is inferred from combined_score and classification
    """

    path = Path(config.candidates_file)

    if not path.exists():
        raise FileNotFoundError(
            f"Candidates file not found: {config.candidates_file}. "
            "Run: python -m execution.live_scheduler"
        )

    if path.stat().st_size == 0:
        raise ValueError("Candidates file is empty.")

    data = pd.read_csv(path)

    if data.empty:
        raise ValueError("Candidates file is empty.")

    required_columns = {
        "instrument_name",
        "option_type",
        "strike",
        "days_to_expiry",
        "market_price_usd",
        "classification",
    }

    missing = required_columns.difference(set(data.columns))

    if missing:
        raise ValueError(f"Candidates file missing columns: {missing}")

    numeric_columns = [
        "spot_price",
        "strike",
        "days_to_expiry",
        "market_price_usd",
        "model_price_usd",
        "price_diff_pct",
        "implied_volatility",
        "implied_volatility_model",
        "implied_volatility_feed",
        "historical_volatility",
        "volatility_spread",
        "cheapness_score",
        "volatility_edge",
        "combined_score",
        "mispricing_score",
    ]

    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    data["classification"] = data["classification"].astype(str).str.lower().str.strip()

    # Case 1:
    # If candidate file has cheapness_score and volatility_edge but no combined_score,
    # create combined_score.
    if "combined_score" not in data.columns:
        if "cheapness_score" in data.columns and "volatility_edge" in data.columns:
            data["combined_score"] = (
                data["cheapness_score"].fillna(0.0).abs()
                + data["volatility_edge"].fillna(0.0).abs()
            )

    # Case 2:
    # If candidate file has mispricing_score but no combined_score,
    # create combined_score from absolute score.
    if "combined_score" not in data.columns and "mispricing_score" in data.columns:
        data["combined_score"] = data["mispricing_score"].abs()

    # Case 3:
    # If candidate file has combined_score but no mispricing_score,
    # infer signed mispricing_score from classification.
    if "mispricing_score" not in data.columns and "combined_score" in data.columns:
        data["combined_score"] = pd.to_numeric(
            data["combined_score"],
            errors="coerce",
        ).fillna(0.0)

        def infer_signed_mispricing_score(row: pd.Series) -> float:
            score = safe_float(row.get("combined_score"))

            if score is None:
                return 0.0

            classification = str(row.get("classification", "")).lower().strip()

            if classification == "cheap":
                return -abs(score)

            if classification == "expensive":
                return abs(score)

            return 0.0

        data["mispricing_score"] = data.apply(
            infer_signed_mispricing_score,
            axis=1,
        )

    if "combined_score" not in data.columns and "mispricing_score" not in data.columns:
        raise ValueError(
            "Candidates file must contain at least one usable score column: "
            "'combined_score', 'mispricing_score', or "
            "'cheapness_score' + 'volatility_edge'."
        )

    # Final cleanup
    if "combined_score" not in data.columns:
        data["combined_score"] = 0.0

    if "mispricing_score" not in data.columns:
        data["mispricing_score"] = 0.0

    data["combined_score"] = pd.to_numeric(
        data["combined_score"],
        errors="coerce",
    ).fillna(0.0)

    data["mispricing_score"] = pd.to_numeric(
        data["mispricing_score"],
        errors="coerce",
    ).fillna(0.0)

    data["market_price_usd"] = pd.to_numeric(
        data["market_price_usd"],
        errors="coerce",
    )

    data = data.dropna(subset=["market_price_usd"])
    data = data[data["market_price_usd"] > 0].copy()

    return data.reset_index(drop=True)


def load_open_positions(config: PaperTraderConfig) -> pd.DataFrame:
    """
    Load current paper open positions.

    If file does not exist or is empty, return empty DataFrame.
    """

    data = load_csv_if_exists(config.positions_file)

    if data.empty:
        return pd.DataFrame()

    numeric_columns = [
        "spot_price",
        "strike",
        "days_to_expiry",
        "market_price_usd",
        "entry_price_usd",
        "quantity",
        "capital_at_risk",
        "current_price_usd",
        "current_value_usd",
        "entry_value_usd",
        "unrealized_pnl",
        "unrealized_pnl_usd",
        "unrealized_pnl_pct",
        "model_price_usd",
        "price_diff_pct",
        "implied_volatility",
        "implied_volatility_model",
        "implied_volatility_feed",
        "historical_volatility",
        "volatility_spread",
        "combined_score",
        "mispricing_score",
    ]

    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    return data.reset_index(drop=True)


def save_open_positions(
    positions: pd.DataFrame,
    config: PaperTraderConfig,
) -> None:
    """
    Save current paper open positions.
    """

    ensure_parent_folder(config.positions_file)
    positions.to_csv(config.positions_file, index=False)


def append_trade_history(
    trades: pd.DataFrame,
    config: PaperTraderConfig,
) -> None:
    """
    Append new trades to trade history CSV.

    Handles missing or empty trade history file safely.
    """

    if trades.empty:
        return

    ensure_parent_folder(config.trade_history_file)

    history_path = Path(config.trade_history_file)

    if history_path.exists() and history_path.stat().st_size > 0:
        try:
            history = pd.read_csv(history_path)
            combined = pd.concat([history, trades], ignore_index=True)
        except pd.errors.EmptyDataError:
            combined = trades.copy()
    else:
        combined = trades.copy()

    combined.to_csv(history_path, index=False)


def filter_trade_candidates(
    candidates: pd.DataFrame,
    open_positions: pd.DataFrame,
    config: PaperTraderConfig,
) -> pd.DataFrame:
    """
    Filter candidates for paper trading.

    Filters:
    - cheap only, if enabled
    - minimum score strength
    - minimum market price
    - avoid duplicate open instruments
    """

    data = candidates.copy()

    if data.empty:
        return data

    data["classification"] = data["classification"].astype(str).str.lower().str.strip()

    if config.only_trade_cheap_options:
        data = data[data["classification"] == "cheap"].copy()

    if data.empty:
        return data

    data["combined_score"] = pd.to_numeric(
        data["combined_score"],
        errors="coerce",
    ).fillna(0.0)

    data["mispricing_score"] = pd.to_numeric(
        data["mispricing_score"],
        errors="coerce",
    ).fillna(0.0)

    data["market_price_usd"] = pd.to_numeric(
        data["market_price_usd"],
        errors="coerce",
    )

    data = data.dropna(subset=["market_price_usd"])
    data = data[data["market_price_usd"] >= float(config.min_market_price_usd)].copy()

    # Signal threshold:
    # combined_score is absolute opportunity strength.
    data = data[
        data["combined_score"] >= float(config.min_abs_mispricing_score)
    ].copy()

    if data.empty:
        return data

    # Avoid duplicate open positions.
    if not open_positions.empty and "instrument_name" in open_positions.columns:
        already_open = set(open_positions["instrument_name"].astype(str).tolist())
        data = data[
            ~data["instrument_name"].astype(str).isin(already_open)
        ].copy()

    if data.empty:
        return data

    # Strongest opportunities first.
    data = data.sort_values(
        by="combined_score",
        ascending=False,
    ).reset_index(drop=True)

    return data


def calculate_position_size(
    cash: float,
    option_price: float,
    max_risk_per_trade: float,
) -> tuple[float, float]:
    """
    Calculate paper position size.

    For a long option, max loss is premium paid.

    Returns
    -------
    tuple[float, float]
        quantity, capital_at_risk
    """

    if cash <= 0:
        return 0.0, 0.0

    if option_price <= 0:
        return 0.0, 0.0

    if max_risk_per_trade <= 0 or max_risk_per_trade > 1:
        return 0.0, 0.0

    risk_amount = cash * max_risk_per_trade

    if risk_amount <= 0:
        return 0.0, 0.0

    if risk_amount > cash:
        risk_amount = cash

    quantity = risk_amount / option_price

    return float(quantity), float(risk_amount)


def open_paper_trades(
    config: PaperTraderConfig | None = None,
) -> pd.DataFrame:
    """
    Open new paper trades from live option candidates.

    Returns
    -------
    pd.DataFrame
        Newly opened paper trades.
    """

    if config is None:
        config = PaperTraderConfig()

    cash = load_cash(config)
    candidates = load_candidates(config)
    open_positions = load_open_positions(config)

    slots_available = int(config.max_positions) - len(open_positions)

    if slots_available <= 0:
        print("No position slots available.")
        return pd.DataFrame()

    trade_candidates = filter_trade_candidates(
        candidates=candidates,
        open_positions=open_positions,
        config=config,
    )

    if trade_candidates.empty:
        print("No trade candidates passed filters.")
        return pd.DataFrame()

    new_trades: list[dict[str, Any]] = []

    for _, row in trade_candidates.head(slots_available).iterrows():
        option_price = safe_float(row.get("market_price_usd"))

        if option_price is None or option_price <= 0:
            continue

        quantity, capital_at_risk = calculate_position_size(
            cash=cash,
            option_price=option_price,
            max_risk_per_trade=config.max_risk_per_trade,
        )

        if quantity <= 0 or capital_at_risk <= 0:
            continue

        if capital_at_risk > cash:
            continue

        cash -= capital_at_risk

        trade = {
            "opened_at": str(pd.Timestamp.now(tz="UTC")),
            "instrument_name": str(row.get("instrument_name")),
            "option_type": str(row.get("option_type")),
            "spot_price": safe_float(row.get("spot_price")),
            "strike": safe_float(row.get("strike")),
            "days_to_expiry": safe_float(row.get("days_to_expiry")),
            "entry_price_usd": float(option_price),
            "quantity": float(quantity),
            "capital_at_risk": float(capital_at_risk),
            "model_price_usd": safe_float(row.get("model_price_usd")),
            "price_diff_pct": safe_float(row.get("price_diff_pct")),
            "implied_volatility": safe_float(row.get("implied_volatility")),
            "implied_volatility_model": safe_float(row.get("implied_volatility_model")),
            "implied_volatility_feed": safe_float(row.get("implied_volatility_feed")),
            "historical_volatility": safe_float(row.get("historical_volatility")),
            "volatility_spread": safe_float(row.get("volatility_spread")),
            "cheapness_score": safe_float(row.get("cheapness_score")),
            "volatility_edge": safe_float(row.get("volatility_edge")),
            "combined_score": safe_float(row.get("combined_score")),
            "mispricing_score": safe_float(row.get("mispricing_score")),
            "classification": str(row.get("classification")),
            "status": "open",
        }

        new_trades.append(trade)

    new_trades_df = pd.DataFrame(new_trades)

    if new_trades_df.empty:
        print("No new paper trades opened.")
        return new_trades_df

    updated_positions = pd.concat(
        [open_positions, new_trades_df],
        ignore_index=True,
    )

    save_open_positions(updated_positions, config)
    append_trade_history(new_trades_df, config)
    save_cash(cash, config)

    print("========== PAPER TRADES OPENED ==========")
    print(f"New trades opened:       {len(new_trades_df)}")
    print(f"Remaining paper cash:    ${cash:,.2f}")
    print(f"Open positions saved to: {config.positions_file}")
    print(f"Trade history saved to:  {config.trade_history_file}")
    print("=========================================")

    display_columns = [
        "instrument_name",
        "option_type",
        "strike",
        "entry_price_usd",
        "quantity",
        "capital_at_risk",
        "combined_score",
        "mispricing_score",
    ]

    available_columns = [
        column for column in display_columns if column in new_trades_df.columns
    ]

    print()
    print(new_trades_df[available_columns].to_string(index=False))

    return new_trades_df


def print_paper_account_summary(
    config: PaperTraderConfig | None = None,
) -> None:
    """
    Print paper account summary.
    """

    if config is None:
        config = PaperTraderConfig()

    cash = load_cash(config)
    positions = load_open_positions(config)

    print("\n========== PAPER ACCOUNT SUMMARY ==========")
    print(f"Paper cash:       ${cash:,.2f}")
    print(f"Open positions:   {len(positions)}")

    if not positions.empty and "capital_at_risk" in positions.columns:
        total_risk = pd.to_numeric(
            positions["capital_at_risk"],
            errors="coerce",
        ).fillna(0.0).sum()

        print(f"Capital at risk:  ${float(total_risk):,.2f}")

    if not positions.empty and "unrealized_pnl_usd" in positions.columns:
        unrealized = pd.to_numeric(
            positions["unrealized_pnl_usd"],
            errors="coerce",
        ).fillna(0.0).sum()

        print(f"Unrealized PnL:   ${float(unrealized):,.2f}")

    print("===========================================")

    if not positions.empty:
        display_columns = [
            "instrument_name",
            "option_type",
            "strike",
            "entry_price_usd",
            "quantity",
            "capital_at_risk",
            "combined_score",
            "mispricing_score",
            "status",
        ]

        available_columns = [
            column for column in display_columns if column in positions.columns
        ]

        print()
        print(positions[available_columns].to_string(index=False))


if __name__ == "__main__":
    paper_config = PaperTraderConfig(
        candidates_file="outputs/live_backtest_candidates.csv",
        positions_file="outputs/paper_open_positions.csv",
        trade_history_file="outputs/paper_trade_history.csv",
        initial_cash_file="outputs/paper_cash.csv",
        initial_cash=10_000.0,
        max_positions=10,
        max_risk_per_trade=0.01,
        only_trade_cheap_options=True,
        min_abs_mispricing_score=0.05,
        min_market_price_usd=5.0,
    )

    open_paper_trades(paper_config)
    print_paper_account_summary(paper_config)