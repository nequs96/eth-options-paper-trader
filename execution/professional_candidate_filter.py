"""Professional dynamic candidate filter for ETH options.

Reads raw candidates, applies invest/do-not-invest gates, enriches candidates
with ETH forward-volatility and Market Confidence Index, and writes accepted and
rejected files.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import math

import pandas as pd

from data.market_data import download_eth_data, get_close_prices
from models.eth_forward_volatility import calculate_volatility_forecast
from models.trend_regime_model import get_trend_regime
from models.market_confidence import calculate_market_confidence
from models.liquidity_model import evaluate_liquidity

RAW_CANDIDATES_FILE = "outputs/live_backtest_candidates.csv"
FILTERED_CANDIDATES_FILE = "outputs/live_backtest_candidates_filtered.csv"
REJECTED_CANDIDATES_FILE = "outputs/live_backtest_candidates_rejected.csv"


@dataclass
class ProfessionalFilterConfig:
    raw_candidates_file: str = RAW_CANDIDATES_FILE
    filtered_candidates_file: str = FILTERED_CANDIDATES_FILE
    rejected_candidates_file: str = REJECTED_CANDIDATES_FILE
    historical_vol_start_date: str = "2023-01-01"
    min_days_to_expiry: float = 3.0
    max_days_to_expiry: float = 45.0
    min_market_price_usd: float = 5.0
    max_bid_ask_spread_pct: float = 0.35
    min_mci_to_accept: float = 0.35
    min_liquidity_score: float = 0.25
    min_edge_score: float = 0.25
    min_delta_abs: float = 0.08
    max_delta_abs: float = 0.75
    max_abs_moneyness_pct: float = 0.35


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
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def extract_expiry_from_instrument(instrument_name: Any) -> str:
    parts = str(instrument_name).split("-")
    return parts[1] if len(parts) >= 2 else ""


def normalize_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    data = candidates.copy()
    numeric_columns = [
        "market_price_usd", "model_price_usd", "theoretical_price", "price_diff_pct",
        "volatility_spread", "days_to_expiry", "bid_ask_spread_pct", "strike",
        "strike_price", "spot_price", "underlying_price_usd", "delta", "gamma", "vega",
        "theta", "open_interest", "mispricing_score", "combined_score", "ensemble_score",
        "implied_volatility", "mark_iv", "iv",
    ]
    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    if "option_type" not in data.columns and "instrument_name" in data.columns:
        data["option_type"] = data["instrument_name"].astype(str).str.split("-").str[-1]
    if "option_type" in data.columns:
        data["option_type"] = data["option_type"].astype(str).str.lower().str.strip().replace({"c": "call", "p": "put"})
    if "classification" in data.columns:
        data["classification"] = data["classification"].astype(str).str.lower().str.strip()
    if "strike" not in data.columns and "strike_price" in data.columns:
        data["strike"] = data["strike_price"]
    if "expiry" not in data.columns and "instrument_name" in data.columns:
        data["expiry"] = data["instrument_name"].apply(extract_expiry_from_instrument)
    if "market_price_usd" not in data.columns and "entry_price_usd" in data.columns:
        data["market_price_usd"] = pd.to_numeric(data["entry_price_usd"], errors="coerce")
    return data


def reject_candidate(row: pd.Series, reason: str) -> dict[str, Any]:
    result = row.to_dict()
    result["decision"] = "reject"
    result["decision_reason"] = reason
    return result


def moneyness_pct(row: pd.Series) -> float:
    strike = safe_float(row.get("strike")) or safe_float(row.get("strike_price")) or 0.0
    spot = safe_float(row.get("spot_price")) or safe_float(row.get("underlying_price_usd")) or 0.0
    if spot <= 0 or strike <= 0:
        return 0.0
    return float((strike - spot) / spot)


def hard_gate(row: pd.Series, cfg: ProfessionalFilterConfig) -> tuple[bool, str]:
    dte = safe_float(row.get("days_to_expiry"))
    price = safe_float(row.get("market_price_usd"))
    spread = safe_float(row.get("bid_ask_spread_pct"))
    option_type = str(row.get("option_type", "")).lower().strip()
    delta = safe_float(row.get("delta"))
    classification = str(row.get("classification", "")).lower().strip()

    if option_type not in {"call", "put"}:
        return False, "invalid_option_type"
    if dte is None or dte < cfg.min_days_to_expiry:
        return False, "dte_too_low"
    if dte > cfg.max_days_to_expiry:
        return False, "dte_too_high"
    if price is None or price < cfg.min_market_price_usd:
        return False, "market_price_too_low"
    if spread is not None and spread > cfg.max_bid_ask_spread_pct:
        return False, "spread_too_wide"
    if classification and classification != "cheap":
        return False, "not_classified_cheap"
    if delta is not None and not (cfg.min_delta_abs <= abs(delta) <= cfg.max_delta_abs):
        return False, "delta_outside_dynamic_band"
    if abs(moneyness_pct(row)) > cfg.max_abs_moneyness_pct:
        return False, "moneyness_too_extreme"
    return True, "passed_hard_gate"


def filter_candidate_file(config: ProfessionalFilterConfig | None = None) -> pd.DataFrame:
    if config is None:
        config = ProfessionalFilterConfig()
    raw = load_csv_if_exists(config.raw_candidates_file)
    ensure_parent_folder(config.filtered_candidates_file)
    ensure_parent_folder(config.rejected_candidates_file)
    if raw.empty:
        pd.DataFrame().to_csv(config.filtered_candidates_file, index=False)
        pd.DataFrame().to_csv(config.rejected_candidates_file, index=False)
        print("No raw candidates to filter.")
        return pd.DataFrame()

    candidates = normalize_candidates(raw)
    try:
        eth = download_eth_data(start_date=config.historical_vol_start_date)
        close = get_close_prices(eth).dropna()
        vol_forecast = calculate_volatility_forecast(close)
    except Exception as error:
        print(f"WARNING: Could not build ETH volatility forecast: {error}")
        vol_forecast = calculate_volatility_forecast(pd.Series(dtype=float))

    try:
        trend_regime = get_trend_regime(start_date=config.historical_vol_start_date)
    except Exception as error:
        print(f"WARNING: Could not load trend regime: {error}")
        trend_regime = None

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for _, row in candidates.iterrows():
        ok, reason = hard_gate(row, config)
        if not ok:
            rejected.append(reject_candidate(row, reason))
            continue
        price = safe_float(row.get("market_price_usd")) or 0.0
        spread = safe_float(row.get("bid_ask_spread_pct")) or 0.0
        oi = safe_float(row.get("open_interest"))
        liquidity = evaluate_liquidity(price, spread, oi, min_price_usd=config.min_market_price_usd, max_spread_pct=config.max_bid_ask_spread_pct)
        if not liquidity.allowed:
            rejected.append(reject_candidate(row, liquidity.reason))
            continue
        mc = calculate_market_confidence(row, vol_forecast, trend_regime)
        enriched = row.to_dict()
        enriched.update({
            "decision": "accept" if not mc.reject_reason else "reject",
            "decision_reason": "accepted_dynamic_mci" if not mc.reject_reason else mc.reject_reason,
            "mci": mc.mci,
            "edge_score": mc.edge_score,
            "regime_score": mc.regime_score,
            "vol_score": mc.vol_score,
            "liquidity_score": mc.liquidity_score,
            "greek_score": mc.greek_score,
            "portfolio_score": mc.portfolio_score,
            "required_price_edge": mc.required_price_edge,
            "required_vol_edge": mc.required_vol_edge,
            "expected_return_hurdle": mc.expected_return_hurdle,
            "mci_reject_reason": mc.reject_reason,
            "forecast_vol": vol_forecast.forecast_vol,
            "vol_expansion_score": vol_forecast.expansion_score,
            "vol_rank": vol_forecast.vol_rank,
            "vol_zscore": vol_forecast.vol_zscore,
            "liquidity_reason": liquidity.reason,
        })
        if mc.reject_reason or mc.mci < config.min_mci_to_accept or mc.liquidity_score < config.min_liquidity_score or mc.edge_score < config.min_edge_score:
            enriched["decision"] = "reject"
            enriched["decision_reason"] = enriched.get("mci_reject_reason") or "mci_quality_below_acceptance_threshold"
            rejected.append(enriched)
        else:
            accepted.append(enriched)

    accepted_df = pd.DataFrame(accepted)
    rejected_df = pd.DataFrame(rejected)
    if not accepted_df.empty:
        accepted_df = accepted_df.sort_values("mci", ascending=False).reset_index(drop=True)
    accepted_df.to_csv(config.filtered_candidates_file, index=False)
    rejected_df.to_csv(config.rejected_candidates_file, index=False)
    print(f"Professional filter complete. Accepted={len(accepted_df)}, Rejected={len(rejected_df)}")
    return accepted_df


if __name__ == "__main__":
    filter_candidate_file()
