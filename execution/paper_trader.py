"""
execution/paper_trader.py

Persistent paper trading module with dynamic portfolio allocation
and confidence-based position sizing.

Core design:
- Professional filter decides which candidates are eligible.
- This module decides whether eligible candidates deserve portfolio capital now.
- max_positions is a hard cap, not a target.
- target_positions is normal desired exposure.
- positions above target require stronger candidate scores.
- position risk is dynamic: 0.1% to 4.0% by default.
- total portfolio open risk is capped.

Paper trading only. No real orders are placed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import math
import pandas as pd


@dataclass
class PaperTraderConfig:
    candidates_file: str = "outputs/live_backtest_candidates_filtered.csv"
    positions_file: str = "outputs/paper_open_positions.csv"
    trade_history_file: str = "outputs/paper_trade_history.csv"
    initial_cash_file: str = "outputs/paper_cash.csv"

    initial_cash: float = 10_000.0

    # Old fallback/default risk. Kept for backward compatibility.
    max_risk_per_trade: float = 0.01

    # New confidence-based position sizing.
    # 0.001 = 0.1%
    # 0.04 = 4.0%
    min_risk_per_trade: float = 0.001
    max_confidence_risk_per_trade: float = 0.04

    # Portfolio-level risk cap.
    # For long options, premium paid is approximately max loss.
    max_total_open_risk_pct: float = 0.25

    min_market_price_usd: float = 5.0
    only_trade_cheap_options: bool = True
    min_abs_mispricing_score: float = 0.05

    # Dynamic allocation.
    # max_positions = hard safety cap.
    # target_positions = normal desired exposure.
    max_positions: int = 30
    target_positions: int = 4
    max_new_positions_per_cycle: int = 2

    # Score thresholds.
    normal_min_score: float = 0.25
    expansion_min_score: float = 0.45
    exceptional_min_score: float = 0.60
    min_relative_to_best_score: float = 0.75

    # Concentration limits.
    max_same_option_type_positions: int = 5
    max_same_expiry_positions: int = 4
    max_same_option_type_and_expiry_positions: int = 3

    # Cash reserve. Avoid allocating the final part of the paper account.
    min_cash_buffer_pct: float = 0.05


def ensure_parent_folder(file_path: str) -> None:
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None

    return float(number)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def load_csv_if_exists(file_path: str) -> pd.DataFrame:
    path = Path(file_path)

    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def load_cash(config: PaperTraderConfig) -> float:
    data = load_csv_if_exists(config.initial_cash_file)

    if data.empty:
        save_cash(config.initial_cash, config)
        return float(config.initial_cash)

    for column in ["cash", "paper_cash", "current_cash"]:
        if column in data.columns and len(data) > 0:
            value = safe_float(data.iloc[-1][column])

            if value is not None:
                return float(value)

    save_cash(config.initial_cash, config)
    return float(config.initial_cash)


def save_cash(cash: float, config: PaperTraderConfig) -> None:
    ensure_parent_folder(config.initial_cash_file)
    pd.DataFrame([{"cash": float(cash)}]).to_csv(
        config.initial_cash_file,
        index=False,
    )


def load_candidates(config: PaperTraderConfig) -> pd.DataFrame:
    return load_csv_if_exists(config.candidates_file)


def load_open_positions(config: PaperTraderConfig) -> pd.DataFrame:
    data = load_csv_if_exists(config.positions_file)

    if data.empty:
        return data

    if "status" in data.columns:
        data = data[data["status"].astype(str).str.lower().eq("open")]

    return data.reset_index(drop=True)


def save_open_positions(
    positions: pd.DataFrame,
    config: PaperTraderConfig,
) -> None:
    ensure_parent_folder(config.positions_file)
    positions.to_csv(config.positions_file, index=False)


def append_trade_history(
    trades: pd.DataFrame,
    config: PaperTraderConfig,
) -> None:
    if trades.empty:
        return

    ensure_parent_folder(config.trade_history_file)

    old = load_csv_if_exists(config.trade_history_file)

    if old.empty:
        trades.to_csv(config.trade_history_file, index=False)
    else:
        pd.concat([old, trades], ignore_index=True).to_csv(
            config.trade_history_file,
            index=False,
        )


def extract_expiry_from_instrument(instrument_name: Any) -> str:
    """
    Extract expiry token from Deribit instrument name.

    Example:
    ETH-22MAY26-2450-C -> 22MAY26
    """

    parts = str(instrument_name).split("-")

    if len(parts) >= 2:
        return parts[1]

    return ""


def normalize_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    data = candidates.copy()

    numeric_columns = [
        "market_price_usd",
        "entry_price_usd",
        "combined_score",
        "ensemble_score",
        "mispricing_score",
        "price_diff_pct",
        "volatility_spread",
        "days_to_expiry",
        "delta",
        "strike",
        "spot_price",
        "bid_ask_spread_pct",
        "open_interest",
    ]

    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    if "classification" in data.columns:
        data["classification"] = (
            data["classification"]
            .astype(str)
            .str.lower()
            .str.strip()
        )

    if "option_type" in data.columns:
        data["option_type"] = (
            data["option_type"]
            .astype(str)
            .str.lower()
            .str.strip()
        )

    if "expiry" not in data.columns and "instrument_name" in data.columns:
        data["expiry"] = data["instrument_name"].apply(
            extract_expiry_from_instrument
        )

    if "market_price_usd" not in data.columns and "entry_price_usd" in data.columns:
        data["market_price_usd"] = data["entry_price_usd"]

    return data


def get_candidate_score(row: pd.Series) -> float:
    """
    Return candidate opportunity strength.

    Priority:
    1. ensemble_score
    2. combined_score
    3. mispricing_score

    Scores are converted to absolute values because some old signals may encode
    cheap options as negative mispricing. For allocation, we need magnitude.
    """

    for column in ["ensemble_score", "combined_score", "mispricing_score"]:
        if column in row.index:
            value = safe_float(row.get(column))

            if value is not None:
                return abs(float(value))

    return 0.0


def calculate_confidence_score(
    candidate_score: float,
    config: PaperTraderConfig,
) -> float:
    """
    Convert candidate score into 0..1 confidence.

    normal_min_score      -> confidence close to 0
    exceptional_min_score -> confidence close to 1
    """

    low = config.normal_min_score
    high = config.exceptional_min_score

    if high <= low:
        return 0.0

    raw_confidence = (candidate_score - low) / (high - low)

    return clamp(raw_confidence, 0.0, 1.0)


def calculate_dynamic_risk_pct(
    candidate_score: float,
    config: PaperTraderConfig,
) -> float:
    """
    Convert confidence into risk percentage.

    Uses a nonlinear confidence curve:
    - average candidates remain small,
    - only very strong candidates approach max risk.
    """

    confidence = calculate_confidence_score(candidate_score, config)

    # Squared confidence makes sizing conservative.
    curved_confidence = confidence ** 2

    risk_pct = (
        config.min_risk_per_trade
        + curved_confidence
        * (config.max_confidence_risk_per_trade - config.min_risk_per_trade)
    )

    return clamp(
        risk_pct,
        config.min_risk_per_trade,
        config.max_confidence_risk_per_trade,
    )


def calculate_open_risk_amount(open_positions: pd.DataFrame) -> float:
    """
    Calculate total capital currently at risk in open positions.

    For long options, premium paid is approximately max possible loss.
    """

    if open_positions.empty:
        return 0.0

    data = open_positions.copy()

    if "status" in data.columns:
        data = data[data["status"].astype(str).str.lower().eq("open")]

    if data.empty:
        return 0.0

    if "capital_at_risk" not in data.columns:
        return 0.0

    risk = pd.to_numeric(
        data["capital_at_risk"],
        errors="coerce",
    ).fillna(0.0)

    return float(risk.sum())


def count_matching_positions(
    open_positions: pd.DataFrame,
    option_type: str | None = None,
    expiry: str | None = None,
) -> int:
    if open_positions.empty:
        return 0

    data = open_positions.copy()

    if "status" in data.columns:
        data = data[data["status"].astype(str).str.lower().eq("open")]

    if option_type is not None and "option_type" in data.columns:
        data = data[
            data["option_type"]
            .astype(str)
            .str.lower()
            .str.strip()
            .eq(option_type.lower().strip())
        ]

    if expiry is not None and "expiry" in data.columns:
        data = data[
            data["expiry"]
            .astype(str)
            .str.strip()
            .eq(str(expiry).strip())
        ]

    return len(data)


def candidate_passes_portfolio_allocation(
    row: pd.Series,
    open_positions: pd.DataFrame,
    config: PaperTraderConfig,
    best_score: float,
) -> tuple[bool, str]:
    current_open_count = len(open_positions)

    if current_open_count >= config.max_positions:
        return False, "max_positions_reached"

    score = get_candidate_score(row)
    relative_score = score / best_score if best_score > 0 else 0.0

    if relative_score < config.min_relative_to_best_score:
        return False, "not_close_enough_to_best_candidate"

    # Normal zone: build toward target.
    if current_open_count < config.target_positions:
        if score < config.normal_min_score:
            return False, "score_below_normal_minimum"

    # Expansion zone: portfolio already has normal exposure.
    elif current_open_count < 8:
        if score < config.expansion_min_score:
            return False, "score_not_strong_enough_to_expand"

    # Large exposure zone: require exceptional candidates only.
    else:
        if score < config.exceptional_min_score:
            return False, "score_not_exceptional_enough"

    option_type = str(row.get("option_type", "")).lower().strip()
    expiry = str(row.get("expiry", "")).strip()

    if option_type:
        same_type_count = count_matching_positions(
            open_positions,
            option_type=option_type,
        )

        if same_type_count >= config.max_same_option_type_positions:
            return False, "too_many_same_option_type_positions"

    if expiry:
        same_expiry_count = count_matching_positions(
            open_positions,
            expiry=expiry,
        )

        if same_expiry_count >= config.max_same_expiry_positions:
            return False, "too_many_same_expiry_positions"

        same_type_expiry_count = count_matching_positions(
            open_positions,
            option_type=option_type,
            expiry=expiry,
        )

        if same_type_expiry_count >= config.max_same_option_type_and_expiry_positions:
            return False, "too_many_same_type_same_expiry_positions"

    return True, "accepted_by_portfolio_allocation"


def filter_trade_candidates(
    candidates: pd.DataFrame,
    open_positions: pd.DataFrame,
    config: PaperTraderConfig,
) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()

    data = normalize_candidates(candidates)

    # Do not reopen the same instrument.
    if "instrument_name" in data.columns and "instrument_name" in open_positions.columns:
        existing_instruments = set(open_positions["instrument_name"].astype(str))
        data = data[~data["instrument_name"].astype(str).isin(existing_instruments)]

    # Long-option strategy: usually trade cheap options only.
    if config.only_trade_cheap_options and "classification" in data.columns:
        data = data[data["classification"].eq("cheap")]

    if "market_price_usd" in data.columns:
        data = data[
            pd.to_numeric(
                data["market_price_usd"],
                errors="coerce",
            )
            >= config.min_market_price_usd
        ]

    if data.empty:
        return pd.DataFrame()

    data["portfolio_candidate_score"] = data.apply(
        get_candidate_score,
        axis=1,
    )

    data = data[
        data["portfolio_candidate_score"] >= config.min_abs_mispricing_score
    ]

    data = data.sort_values(
        "portfolio_candidate_score",
        ascending=False,
    ).reset_index(drop=True)

    if data.empty:
        return pd.DataFrame()

    current_open_count = len(open_positions)

    if current_open_count >= config.max_positions:
        return pd.DataFrame()

    best_score = float(data["portfolio_candidate_score"].max())

    if current_open_count < config.target_positions:
        available_slots = config.target_positions - current_open_count
    else:
        available_slots = config.max_positions - current_open_count

    available_slots = min(
        available_slots,
        config.max_new_positions_per_cycle,
    )

    selected: list[dict[str, Any]] = []
    exposure_view = open_positions.copy()

    for _, row in data.iterrows():
        if len(selected) >= available_slots:
            break

        allowed, reason = candidate_passes_portfolio_allocation(
            row=row,
            open_positions=exposure_view,
            config=config,
            best_score=best_score,
        )

        if not allowed:
            continue

        candidate = row.to_dict()
        candidate["portfolio_allocation_reason"] = reason
        selected.append(candidate)

        # Important:
        # Treat selected same-cycle candidates as temporary exposure so that
        # the bot does not pick many highly similar positions at once.
        exposure_view = pd.concat(
            [exposure_view, pd.DataFrame([candidate])],
            ignore_index=True,
        )

    return pd.DataFrame(selected)


def calculate_position_size(
    cash: float,
    option_price: float,
    risk_pct: float,
) -> tuple[float, float]:
    """
    Calculate paper position size from dynamic risk percentage.
    """

    if cash <= 0 or option_price <= 0 or risk_pct <= 0:
        return 0.0, 0.0

    risk_amount = cash * risk_pct
    quantity = risk_amount / option_price

    return float(quantity), float(risk_amount)


def open_paper_trades(config: PaperTraderConfig | None = None) -> pd.DataFrame:
    if config is None:
        config = PaperTraderConfig()

    candidates = load_candidates(config)
    open_positions = load_open_positions(config)
    cash = load_cash(config)

    selected = filter_trade_candidates(
        candidates=candidates,
        open_positions=open_positions,
        config=config,
    )

    if selected.empty:
        print("No candidates passed dynamic portfolio allocation.")
        return pd.DataFrame()

    now = pd.Timestamp.utcnow().isoformat()
    min_cash_buffer = config.initial_cash * config.min_cash_buffer_pct
    max_total_open_risk = config.initial_cash * config.max_total_open_risk_pct

    new_positions: list[dict[str, Any]] = []

    # Keep this updated inside the loop so same-cycle risk does not exceed cap.
    current_open_risk = calculate_open_risk_amount(open_positions)

    for _, row in selected.iterrows():
        option_price = (
            safe_float(row.get("market_price_usd"))
            or safe_float(row.get("entry_price_usd"))
        )

        if option_price is None or option_price <= 0:
            continue

        if cash <= min_cash_buffer:
            print("Cash buffer reached. No more positions opened this cycle.")
            break

        candidate_score = safe_float(row.get("portfolio_candidate_score"))

        if candidate_score is None:
            candidate_score = get_candidate_score(row)

        confidence_score = calculate_confidence_score(
            candidate_score=candidate_score,
            config=config,
        )

        dynamic_risk_pct = calculate_dynamic_risk_pct(
            candidate_score=candidate_score,
            config=config,
        )

        remaining_risk_capacity = max_total_open_risk - current_open_risk

        if remaining_risk_capacity <= 0:
            print("Portfolio total open risk cap reached.")
            break

        quantity, capital_at_risk = calculate_position_size(
            cash=cash,
            option_price=option_price,
            risk_pct=dynamic_risk_pct,
        )

        # Do not exceed total portfolio risk capacity.
        if capital_at_risk > remaining_risk_capacity:
            capital_at_risk = remaining_risk_capacity
            quantity = capital_at_risk / option_price

        # Do not exceed cash.
        if capital_at_risk > cash:
            capital_at_risk = cash
            quantity = capital_at_risk / option_price

        if quantity <= 0 or capital_at_risk <= 0:
            continue

        position = row.to_dict()

        position.update(
            {
                "opened_at": now,
                "updated_at": now,
                "status": "open",

                "entry_price_usd": float(option_price),
                "current_price_usd": float(option_price),
                "highest_price_usd": float(option_price),
                "highest_profit_pct": 0.0,
                "trailing_stop_price_usd": 0.0,

                "quantity": float(quantity),
                "capital_at_risk": float(capital_at_risk),
                "current_value_usd": float(capital_at_risk),
                "unrealized_pnl_usd": 0.0,
                "unrealized_pnl_pct": 0.0,

                "portfolio_candidate_score": float(candidate_score),
                "confidence_score": float(confidence_score),
                "dynamic_risk_pct": float(dynamic_risk_pct),
                "dynamic_risk_pct_display": f"{dynamic_risk_pct:.2%}",
                "max_total_open_risk_pct": float(config.max_total_open_risk_pct),
                "portfolio_open_risk_before_trade": float(current_open_risk),
                "portfolio_open_risk_after_trade": float(
                    current_open_risk + capital_at_risk
                ),
            }
        )

        cash -= capital_at_risk
        current_open_risk += capital_at_risk
        new_positions.append(position)

    if not new_positions:
        print("Candidates were selected, but no positions could be sized/opened.")
        return pd.DataFrame()

    updated_positions = pd.concat(
        [open_positions, pd.DataFrame(new_positions)],
        ignore_index=True,
    )

    save_open_positions(updated_positions, config)
    save_cash(cash, config)

    opened = pd.DataFrame(new_positions)

    print(f"Opened {len(opened)} new paper position(s).")
    print(f"Current paper cash: ${cash:,.2f}")
    print(f"Current open risk:  ${current_open_risk:,.2f}")
    print(f"Max open risk cap:  ${max_total_open_risk:,.2f}")

    if "dynamic_risk_pct_display" in opened.columns:
        print("\n--- NEW POSITION SIZING ---")
        print(
            opened[
                [
                    "instrument_name",
                    "option_type",
                    "portfolio_candidate_score",
                    "confidence_score",
                    "dynamic_risk_pct_display",
                    "capital_at_risk",
                ]
            ].to_string(index=False)
        )

    return opened


def print_paper_account_summary(config: PaperTraderConfig | None = None) -> None:
    if config is None:
        config = PaperTraderConfig()

    cash = load_cash(config)
    positions = load_open_positions(config)
    open_risk = calculate_open_risk_amount(positions)
    max_total_open_risk = config.initial_cash * config.max_total_open_risk_pct

    print("\n========== PAPER ACCOUNT SUMMARY ==========")
    print(f"Paper cash:          ${cash:,.2f}")
    print(f"Open positions:      {len(positions)}")
    print(f"Target positions:    {config.target_positions}")
    print(f"Max positions:       {config.max_positions}")
    print(f"Open risk:           ${open_risk:,.2f}")
    print(f"Max open risk cap:   ${max_total_open_risk:,.2f}")
    print(f"Min risk/trade:      {config.min_risk_per_trade:.2%}")
    print(f"Max risk/trade:      {config.max_confidence_risk_per_trade:.2%}")
    print("===========================================")


if __name__ == "__main__":
    open_paper_trades(PaperTraderConfig())