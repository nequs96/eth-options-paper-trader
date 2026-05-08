"""
execution/professional_candidate_filter.py

Professional invest / do-not-invest decision gate for ETH options paper trading.

This module reads raw candidates from:
- outputs/live_backtest_candidates.csv

It applies mandatory hard gates:
- trend regime gate
- volatility expansion gate
- liquidity gate
- expiry gate
- spread gate
- moneyness gate
- delta gate, if available
- mispricing strength gate
- volatility edge gate

It writes:
- outputs/live_backtest_candidates_filtered.csv
- outputs/live_backtest_candidates_rejected.csv

Only candidates written to live_backtest_candidates_filtered.csv are allowed
to reach the paper trader.

Important:
This does not guarantee profitability.
It only prevents low-quality setups from being traded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import math
import pandas as pd

from models.trend_regime_model import get_trend_regime
from models.liquidity_model import evaluate_liquidity


RAW_CANDIDATES_FILE = "outputs/live_backtest_candidates.csv"
FILTERED_CANDIDATES_FILE = "outputs/live_backtest_candidates_filtered.csv"
REJECTED_CANDIDATES_FILE = "outputs/live_backtest_candidates_rejected.csv"


@dataclass
class ProfessionalFilterConfig:
    """
    Configuration for professional invest / no-invest logic.
    """

    raw_candidates_file: str = RAW_CANDIDATES_FILE
    filtered_candidates_file: str = FILTERED_CANDIDATES_FILE
    rejected_candidates_file: str = REJECTED_CANDIDATES_FILE

    historical_vol_start_date: str = "2023-01-01"

    # Main strategy direction
    only_trade_cheap_options: bool = True

    # Minimum signal strength
    min_combined_score: float = 0.15

    # A cheap option should be meaningfully below model value.
    # Example: -0.12 means market price is at least 12% below model value.
    max_price_diff_pct: float = -0.12

    # For long options, we prefer IV below historical vol.
    # Example: -0.08 means IV is at least 8 vol points below historical vol.
    max_volatility_spread: float = -0.08

    # Liquidity / execution
    min_market_price_usd: float = 10.0
    max_bid_ask_spread_pct: float = 0.20

    # Expiry window
    min_days_to_expiry: float = 10.0
    max_days_to_expiry: float = 30.0

    # Moneyness range: strike / spot
    call_min_moneyness: float = 0.90
    call_max_moneyness: float = 1.15
    put_min_moneyness: float = 0.85
    put_max_moneyness: float = 1.10

    # Delta filter, only used if candidate has delta
    min_abs_delta: float = 0.25
    max_abs_delta: float = 0.65

    # If True, no trades when volatility is not expanding
    require_volatility_expansion: bool = False

    # If True, no trades in hostile regime
    block_hostile_regime: bool = False


def ensure_parent_folder(file_path: str) -> None:
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def load_csv_if_exists(file_path: str) -> pd.DataFrame:
    path = Path(file_path)

    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def safe_float(value: Any) -> float | None:
    if value is None:
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if pd.isna(number) or not math.isfinite(number):
        return None

    return float(number)


def normalize_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize candidate columns so the decision logic can safely read them.
    """

    data = candidates.copy()

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
        "mispricing_score",
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

    # Make score columns compatible.
    if "combined_score" not in data.columns:
        if "mispricing_score" in data.columns:
            data["combined_score"] = data["mispricing_score"].abs()
        elif "cheapness_score" in data.columns and "volatility_edge" in data.columns:
            data["combined_score"] = (
                data["cheapness_score"].abs().fillna(0.0)
                + data["volatility_edge"].abs().fillna(0.0)
            )
        else:
            data["combined_score"] = 0.0

    if "bid_ask_spread_pct" not in data.columns:
        data["bid_ask_spread_pct"] = None

    if "open_interest" not in data.columns:
        data["open_interest"] = None

    if "delta" not in data.columns:
        data["delta"] = None

    return data


def reject_candidate(row: pd.Series, reason: str) -> dict[str, Any]:
    """
    Convert rejected candidate row into dictionary with rejection reason.
    """

    result = row.to_dict()
    result["invest_decision"] = "do_not_invest"
    result["invest_allowed"] = False
    result["rejection_reason"] = reason

    return result


def accept_candidate(
    row: pd.Series,
    liquidity_allowed: bool,
    liquidity_reason: str,
    liquidity_score: float,
    regime_bullish: bool,
    regime_bearish: bool,
    regime_volatility_expanding: bool,
    regime_hostile: bool,
) -> dict[str, Any]:
    """
    Convert accepted candidate row into dictionary with decision metadata.
    """

    result = row.to_dict()
    result["invest_decision"] = "invest"
    result["invest_allowed"] = True
    result["rejection_reason"] = ""

    result["liquidity_allowed"] = liquidity_allowed
    result["liquidity_reason"] = liquidity_reason
    result["liquidity_score"] = liquidity_score

    result["regime_bullish"] = regime_bullish
    result["regime_bearish"] = regime_bearish
    result["regime_volatility_expanding"] = regime_volatility_expanding
    result["regime_hostile"] = regime_hostile

    return result


def decide_invest_or_not(
    row: pd.Series,
    config: ProfessionalFilterConfig,
    trend_regime: Any,
) -> tuple[bool, str, dict[str, Any]]:
    """
    Final invest / do-not-invest decision.

    Returns
    -------
    tuple[bool, str, dict]
        allowed, reason, metadata
    """

    classification = str(row.get("classification", "")).lower().strip()
    option_type = str(row.get("option_type", "")).lower().strip()

    spot_price = safe_float(row.get("spot_price"))
    strike = safe_float(row.get("strike"))
    days_to_expiry = safe_float(row.get("days_to_expiry"))
    market_price = safe_float(row.get("market_price_usd"))
    price_diff_pct = safe_float(row.get("price_diff_pct"))
    volatility_spread = safe_float(row.get("volatility_spread"))
    combined_score = safe_float(row.get("combined_score"))
    bid_ask_spread_pct = safe_float(row.get("bid_ask_spread_pct"))
    open_interest = safe_float(row.get("open_interest"))
    delta = safe_float(row.get("delta"))

    regime_bullish = bool(getattr(trend_regime, "bullish", False))
    regime_bearish = bool(getattr(trend_regime, "bearish", False))
    regime_volatility_expanding = bool(
        getattr(trend_regime, "volatility_expanding", False)
    )
    regime_hostile = bool(getattr(trend_regime, "hostile", False))

    metadata = {
        "regime_bullish": regime_bullish,
        "regime_bearish": regime_bearish,
        "regime_volatility_expanding": regime_volatility_expanding,
        "regime_hostile": regime_hostile,
        "liquidity_allowed": False,
        "liquidity_reason": "",
        "liquidity_score": 0.0,
    }

    # ------------------------------------------------------------
    # 1. Basic candidate validity
    # ------------------------------------------------------------

    if config.only_trade_cheap_options and classification != "cheap":
        return False, "not_cheap", metadata

    if option_type not in {"call", "put"}:
        return False, "invalid_option_type", metadata

    if spot_price is None or spot_price <= 0:
        return False, "invalid_spot_price", metadata

    if strike is None or strike <= 0:
        return False, "invalid_strike", metadata

    if days_to_expiry is None:
        return False, "invalid_days_to_expiry", metadata

    if market_price is None or market_price <= 0:
        return False, "invalid_market_price", metadata

    # ------------------------------------------------------------
    # 2. Regime hard gates
    # ------------------------------------------------------------

    if config.block_hostile_regime and regime_hostile:
        return False, "hostile_market_regime", metadata

    if config.require_volatility_expansion and not regime_volatility_expanding:
        return False, "volatility_not_expanding", metadata

    if option_type == "call" and not regime_bullish:
        return False, "call_rejected_not_bullish_regime", metadata

    if option_type == "put" and not regime_bearish:
        return False, "put_rejected_not_bearish_regime", metadata

    # ------------------------------------------------------------
    # 3. Liquidity hard gate
    # ------------------------------------------------------------

    liquidity_decision = evaluate_liquidity(
        market_price_usd=market_price,
        bid_ask_spread_pct=bid_ask_spread_pct,
        open_interest=open_interest,
        min_price_usd=config.min_market_price_usd,
        max_spread_pct=config.max_bid_ask_spread_pct,
    )

    metadata["liquidity_allowed"] = bool(liquidity_decision.allowed)
    metadata["liquidity_reason"] = str(liquidity_decision.reason)
    metadata["liquidity_score"] = float(liquidity_decision.execution_score)

    if not liquidity_decision.allowed:
        return False, f"liquidity_blocked:{liquidity_decision.reason}", metadata

    # ------------------------------------------------------------
    # 4. Expiry hard gate
    # ------------------------------------------------------------

    if days_to_expiry < config.min_days_to_expiry:
        return False, "too_close_to_expiry", metadata

    if days_to_expiry > config.max_days_to_expiry:
        return False, "too_far_to_expiry", metadata

    # ------------------------------------------------------------
    # 5. Mispricing / edge hard gates
    # ------------------------------------------------------------

    if combined_score is None or combined_score < config.min_combined_score:
        return False, "combined_score_too_low", metadata

    if price_diff_pct is None or price_diff_pct > config.max_price_diff_pct:
        return False, "price_discount_not_strong_enough", metadata

    if volatility_spread is None or volatility_spread > config.max_volatility_spread:
        return False, "volatility_edge_not_strong_enough", metadata

    # ------------------------------------------------------------
    # 6. Moneyness hard gate
    # ------------------------------------------------------------

    moneyness = strike / spot_price

    if option_type == "call":
        if not (
            config.call_min_moneyness
            <= moneyness
            <= config.call_max_moneyness
        ):
            return False, "call_moneyness_out_of_range", metadata

    if option_type == "put":
        if not (
            config.put_min_moneyness
            <= moneyness
            <= config.put_max_moneyness
        ):
            return False, "put_moneyness_out_of_range", metadata

    # ------------------------------------------------------------
    # 7. Delta hard gate, only if delta exists
    # ------------------------------------------------------------

    if delta is not None:
        abs_delta = abs(delta)

        if abs_delta < config.min_abs_delta:
            return False, "delta_too_low", metadata

        if abs_delta > config.max_abs_delta:
            return False, "delta_too_high", metadata

    return True, "invest", metadata


def filter_candidate_file(
    config: ProfessionalFilterConfig | None = None,
) -> pd.DataFrame:
    """
    Apply final invest / do-not-invest logic to raw candidate file.
    """

    if config is None:
        config = ProfessionalFilterConfig()

    raw_candidates = load_csv_if_exists(config.raw_candidates_file)

    if raw_candidates.empty:
        ensure_parent_folder(config.filtered_candidates_file)

        empty = pd.DataFrame()
        empty.to_csv(config.filtered_candidates_file, index=False)
        empty.to_csv(config.rejected_candidates_file, index=False)

        print("No raw candidates found.")
        return empty

    candidates = normalize_candidates(raw_candidates)

    required_columns = {
        "instrument_name",
        "option_type",
        "spot_price",
        "strike",
        "days_to_expiry",
        "market_price_usd",
        "classification",
        "combined_score",
        "price_diff_pct",
        "volatility_spread",
    }

    missing = required_columns.difference(set(candidates.columns))

    if missing:
        raise ValueError(f"Candidate file missing required columns: {missing}")

    trend_regime = get_trend_regime(
        start_date=config.historical_vol_start_date,
    )

    accepted_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []

    for _, row in candidates.iterrows():
        allowed, reason, metadata = decide_invest_or_not(
            row=row,
            config=config,
            trend_regime=trend_regime,
        )

        if allowed:
            accepted_rows.append(
                accept_candidate(
                    row=row,
                    liquidity_allowed=bool(metadata["liquidity_allowed"]),
                    liquidity_reason=str(metadata["liquidity_reason"]),
                    liquidity_score=float(metadata["liquidity_score"]),
                    regime_bullish=bool(metadata["regime_bullish"]),
                    regime_bearish=bool(metadata["regime_bearish"]),
                    regime_volatility_expanding=bool(
                        metadata["regime_volatility_expanding"]
                    ),
                    regime_hostile=bool(metadata["regime_hostile"]),
                )
            )
        else:
            rejected = reject_candidate(row=row, reason=reason)
            rejected["liquidity_allowed"] = bool(metadata["liquidity_allowed"])
            rejected["liquidity_reason"] = str(metadata["liquidity_reason"])
            rejected["liquidity_score"] = float(metadata["liquidity_score"])
            rejected["regime_bullish"] = bool(metadata["regime_bullish"])
            rejected["regime_bearish"] = bool(metadata["regime_bearish"])
            rejected["regime_volatility_expanding"] = bool(
                metadata["regime_volatility_expanding"]
            )
            rejected["regime_hostile"] = bool(metadata["regime_hostile"])
            rejected_rows.append(rejected)

    accepted = pd.DataFrame(accepted_rows)
    rejected = pd.DataFrame(rejected_rows)

    if not accepted.empty:
        accepted = accepted.sort_values(
            by=["combined_score", "market_price_usd"],
            ascending=[False, False],
        ).reset_index(drop=True)

    ensure_parent_folder(config.filtered_candidates_file)

    accepted.to_csv(config.filtered_candidates_file, index=False)
    rejected.to_csv(config.rejected_candidates_file, index=False)

    print("========== INVEST / DO-NOT-INVEST FILTER ==========")
    print(f"Raw candidates:       {len(candidates)}")
    print(f"Invest candidates:    {len(accepted)}")
    print(f"Rejected candidates:  {len(rejected)}")
    print("---------------------------------------------------")
    print(f"Regime bullish:       {bool(getattr(trend_regime, 'bullish', False))}")
    print(f"Regime bearish:       {bool(getattr(trend_regime, 'bearish', False))}")
    print(
        f"Vol expanding:        "
        f"{bool(getattr(trend_regime, 'volatility_expanding', False))}"
    )
    print(f"Regime hostile:       {bool(getattr(trend_regime, 'hostile', False))}")
    print("---------------------------------------------------")
    print(f"Saved invest file:    {config.filtered_candidates_file}")
    print(f"Saved rejected file:  {config.rejected_candidates_file}")
    print("===================================================")

    if not rejected.empty and "rejection_reason" in rejected.columns:
        print("\nTop rejection reasons:")
        print(
            rejected["rejection_reason"]
            .value_counts()
            .head(10)
            .to_string()
        )

    return accepted


if __name__ == "__main__":
    filter_candidate_file()