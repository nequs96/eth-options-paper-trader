"""Persistent dynamic paper trading module for ETH options.

This replacement integrates:
- Market Confidence Index (MCI),
- robust dynamic risk sizing,
- per-position dynamic exit parameters,
- lower ETH-appropriate risk caps.

Paper trading only. No real orders are placed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import math

import pandas as pd

from models.market_confidence import MarketConfidence
from execution.dynamic_risk_sizing import DynamicRiskConfig, calculate_open_risk_pct, size_position
from execution.dynamic_exit_config import build_dynamic_exit_plan, exit_plan_columns


@dataclass
class PaperTraderConfig:
    candidates_file: str = "outputs/live_backtest_candidates_filtered.csv"
    positions_file: str = "outputs/paper_open_positions.csv"
    trade_history_file: str = "outputs/paper_trade_history.csv"
    initial_cash_file: str = "outputs/paper_cash.csv"
    initial_cash: float = 10_000.0

    min_risk_per_trade: float = 0.001
    normal_max_risk_per_trade: float = 0.0125
    exceptional_max_risk_per_trade: float = 0.020
    max_confidence_risk_per_trade: float = 0.020  # backward-compatible alias
    max_total_open_risk_pct: float = 0.10

    min_market_price_usd: float = 5.0
    only_trade_cheap_options: bool = True
    min_abs_mispricing_score: float = 0.05
    min_mci_to_trade: float = 0.35

    max_positions: int = 8
    target_positions: int = 3
    max_new_positions_per_cycle: int = 2
    normal_min_score: float = 0.30
    expansion_min_score: float = 0.50
    exceptional_min_score: float = 0.70
    min_relative_to_best_score: float = 0.80

    max_same_option_type_positions: int = 4
    max_same_expiry_positions: int = 3
    max_same_option_type_and_expiry_positions: int = 2
    min_cash_buffer_pct: float = 0.05


def ensure_parent_folder(file_path: str) -> None:
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def numeric_series(data: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in data.columns:
        return pd.to_numeric(data[column], errors="coerce").fillna(default)
    return pd.Series([default] * len(data), index=data.index, dtype=float)


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
    pd.DataFrame([{"cash": float(cash)}]).to_csv(config.initial_cash_file, index=False)


def load_candidates(config: PaperTraderConfig) -> pd.DataFrame:
    return load_csv_if_exists(config.candidates_file)


def load_open_positions(config: PaperTraderConfig) -> pd.DataFrame:
    data = load_csv_if_exists(config.positions_file)
    if data.empty:
        return data
    if "status" in data.columns:
        data = data[data["status"].astype(str).str.lower().eq("open")]
    return data.reset_index(drop=True)


def save_open_positions(positions: pd.DataFrame, config: PaperTraderConfig) -> None:
    ensure_parent_folder(config.positions_file)
    positions.to_csv(config.positions_file, index=False)


def append_trade_history(trades: pd.DataFrame, config: PaperTraderConfig) -> None:
    if trades.empty:
        return
    ensure_parent_folder(config.trade_history_file)
    old = load_csv_if_exists(config.trade_history_file)
    if old.empty:
        trades.to_csv(config.trade_history_file, index=False)
    else:
        pd.concat([old, trades], ignore_index=True).to_csv(config.trade_history_file, index=False)


def extract_expiry_from_instrument(instrument_name: Any) -> str:
    parts = str(instrument_name).split("-")
    return parts[1] if len(parts) >= 2 else ""


def normalize_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    data = candidates.copy()
    numeric_columns = [
        "market_price_usd", "entry_price_usd", "combined_score", "ensemble_score",
        "mispricing_score", "portfolio_candidate_score", "price_diff_pct", "volatility_spread",
        "days_to_expiry", "delta", "gamma", "vega", "theta", "strike", "strike_price",
        "spot_price", "bid_ask_spread_pct", "open_interest", "mci", "edge_score",
        "regime_score", "vol_score", "liquidity_score", "greek_score", "portfolio_score",
    ]
    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    if "classification" in data.columns:
        data["classification"] = data["classification"].astype(str).str.lower().str.strip()
    if "option_type" in data.columns:
        data["option_type"] = data["option_type"].astype(str).str.lower().str.strip()
    if "strike" not in data.columns and "strike_price" in data.columns:
        data["strike"] = data["strike_price"]
    if "expiry" not in data.columns and "instrument_name" in data.columns:
        data["expiry"] = data["instrument_name"].apply(extract_expiry_from_instrument)
    if "market_price_usd" not in data.columns and "entry_price_usd" in data.columns:
        data["market_price_usd"] = data["entry_price_usd"]
    return data


def get_candidate_score(row: pd.Series) -> float:
    for column in ["mci", "portfolio_candidate_score", "ensemble_score", "combined_score", "mispricing_score"]:
        if column in row.index:
            value = safe_float(row.get(column))
            if value is not None:
                return abs(float(value))
    return 0.0


def open_only(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return data
    result = data.copy()
    if "status" in result.columns:
        result = result[result["status"].astype(str).str.lower().eq("open")]
    return result


def calculate_open_risk_amount(open_positions: pd.DataFrame) -> float:
    data = open_only(open_positions)
    if data.empty:
        return 0.0
    return float(numeric_series(data, "capital_at_risk", 0.0).sum())


def calculate_open_position_value(open_positions: pd.DataFrame) -> float:
    data = open_only(open_positions)
    if data.empty:
        return 0.0
    values = numeric_series(data, "current_value_usd", 0.0)
    if float(values.sum()) > 0:
        return float(values.sum())
    return float((numeric_series(data, "current_price_usd", 0.0) * numeric_series(data, "quantity", 0.0)).sum())


def calculate_unrealized_pnl_amount(open_positions: pd.DataFrame) -> float:
    data = open_only(open_positions)
    if data.empty:
        return 0.0
    pnl = numeric_series(data, "unrealized_pnl_usd", 0.0)
    if abs(float(pnl.sum())) > 0:
        return float(pnl.sum())
    return float(((numeric_series(data, "current_price_usd", 0.0) - numeric_series(data, "entry_price_usd", 0.0)) * numeric_series(data, "quantity", 0.0)).sum())


def count_matching_positions(open_positions: pd.DataFrame, option_type: str | None = None, expiry: str | None = None) -> int:
    data = open_only(open_positions)
    if data.empty:
        return 0
    if option_type is not None and "option_type" in data.columns:
        data = data[data["option_type"].astype(str).str.lower().str.strip().eq(option_type.lower().strip())]
    if expiry is not None and "expiry" in data.columns:
        data = data[data["expiry"].astype(str).str.strip().eq(str(expiry).strip())]
    return int(len(data))


def candidate_passes_portfolio_allocation(row: pd.Series, open_positions: pd.DataFrame, config: PaperTraderConfig, best_score: float) -> tuple[bool, str]:
    current_open_count = len(open_only(open_positions))
    if current_open_count >= config.max_positions:
        return False, "max_positions_reached"

    score = get_candidate_score(row)
    relative_score = score / best_score if best_score > 0 else 0.0
    if relative_score < config.min_relative_to_best_score:
        return False, "not_close_enough_to_best_candidate"
    if score < config.min_mci_to_trade and "mci" in row.index:
        return False, "mci_below_trade_minimum"
    if current_open_count < config.target_positions and score < config.normal_min_score:
        return False, "score_below_normal_minimum"
    if config.target_positions <= current_open_count < 6 and score < config.expansion_min_score:
        return False, "score_not_strong_enough_to_expand"
    if current_open_count >= 6 and score < config.exceptional_min_score:
        return False, "score_not_exceptional_enough"

    option_type = str(row.get("option_type", "")).lower().strip()
    expiry = str(row.get("expiry", "")).strip()
    if option_type and count_matching_positions(open_positions, option_type=option_type) >= config.max_same_option_type_positions:
        return False, "too_many_same_option_type_positions"
    if expiry:
        if count_matching_positions(open_positions, expiry=expiry) >= config.max_same_expiry_positions:
            return False, "too_many_same_expiry_positions"
        if count_matching_positions(open_positions, option_type=option_type, expiry=expiry) >= config.max_same_option_type_and_expiry_positions:
            return False, "too_many_same_type_same_expiry_positions"
    return True, "accepted_by_portfolio_allocation"


def filter_trade_candidates(candidates: pd.DataFrame, open_positions: pd.DataFrame, config: PaperTraderConfig) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    data = normalize_candidates(candidates)
    if "instrument_name" in data.columns and "instrument_name" in open_positions.columns:
        existing = set(open_positions["instrument_name"].astype(str))
        data = data[~data["instrument_name"].astype(str).isin(existing)]
    if config.only_trade_cheap_options and "classification" in data.columns:
        data = data[data["classification"].eq("cheap")]
    if "mci_reject_reason" in data.columns:
        data = data[data["mci_reject_reason"].fillna("").astype(str).eq("")]
    if "market_price_usd" in data.columns:
        data = data[pd.to_numeric(data["market_price_usd"], errors="coerce") >= config.min_market_price_usd]
    if data.empty:
        return pd.DataFrame()

    data["portfolio_candidate_score"] = data.apply(get_candidate_score, axis=1)
    data = data[data["portfolio_candidate_score"] >= config.min_abs_mispricing_score]
    data = data.sort_values("portfolio_candidate_score", ascending=False).reset_index(drop=True)
    if data.empty or len(open_only(open_positions)) >= config.max_positions:
        return pd.DataFrame()

    best_score = float(data["portfolio_candidate_score"].max())
    current_count = len(open_only(open_positions))
    slots = config.target_positions - current_count if current_count < config.target_positions else config.max_positions - current_count
    slots = min(max(slots, 0), config.max_new_positions_per_cycle)

    selected: list[dict[str, Any]] = []
    exposure_view = open_positions.copy()
    for _, row in data.iterrows():
        if len(selected) >= slots:
            break
        allowed, reason = candidate_passes_portfolio_allocation(row, exposure_view, config, best_score)
        if not allowed:
            continue
        candidate = row.to_dict()
        candidate["portfolio_allocation_reason"] = reason
        selected.append(candidate)
        exposure_view = pd.concat([exposure_view, pd.DataFrame([candidate])], ignore_index=True)
    return pd.DataFrame(selected)


def market_confidence_from_row(row: pd.Series) -> MarketConfidence:
    return MarketConfidence(
        mci=safe_float(row.get("mci")) or 0.0,
        edge_score=safe_float(row.get("edge_score")) or 0.0,
        regime_score=safe_float(row.get("regime_score")) or 0.0,
        vol_score=safe_float(row.get("vol_score")) or 0.0,
        liquidity_score=safe_float(row.get("liquidity_score")) or 0.0,
        greek_score=safe_float(row.get("greek_score")) or 0.0,
        portfolio_score=safe_float(row.get("portfolio_score")) or 0.0,
        required_price_edge=safe_float(row.get("required_price_edge")) or 0.0,
        required_vol_edge=safe_float(row.get("required_vol_edge")) or 0.0,
        expected_return_hurdle=safe_float(row.get("expected_return_hurdle")) or 0.0,
        reject_reason=str(row.get("mci_reject_reason", "") or ""),
    )


def open_paper_trades(config: PaperTraderConfig | None = None, current_drawdown: float = 0.0) -> pd.DataFrame:
    if config is None:
        config = PaperTraderConfig()
    candidates = load_candidates(config)
    open_positions = load_open_positions(config)
    cash = load_cash(config)
    selected = filter_trade_candidates(candidates, open_positions, config)
    if selected.empty:
        print("No candidates passed dynamic portfolio allocation.")
        return pd.DataFrame()

    risk_cfg = DynamicRiskConfig(
        min_risk_per_trade=config.min_risk_per_trade,
        normal_max_risk_per_trade=config.normal_max_risk_per_trade,
        exceptional_max_risk_per_trade=config.exceptional_max_risk_per_trade,
        max_total_open_risk_pct=config.max_total_open_risk_pct,
        min_mci_to_trade=config.min_mci_to_trade,
        min_cash_buffer_pct=config.min_cash_buffer_pct,
    )

    now = pd.Timestamp.utcnow().isoformat()
    current_open_risk_pct = calculate_open_risk_pct(open_positions, config.initial_cash)
    new_positions: list[dict[str, Any]] = []

    for _, row in selected.iterrows():
        option_price = safe_float(row.get("market_price_usd")) or safe_float(row.get("entry_price_usd"))
        if option_price is None or option_price <= 0:
            continue
        mc = market_confidence_from_row(row)
        risk_decision = size_position(cash, config.initial_cash, option_price, mc, current_drawdown, current_open_risk_pct, risk_cfg)
        if not risk_decision.allowed:
            print(f"Skipped {row.get('instrument_name', '')}: {risk_decision.reason}")
            continue

        days_to_expiry = safe_float(row.get("days_to_expiry")) or 0.0
        exit_plan = build_dynamic_exit_plan(mc, days_to_expiry)
        position = row.to_dict()
        position.update({
            "opened_at": now,
            "updated_at": now,
            "status": "open",
            "entry_price_usd": float(option_price),
            "current_price_usd": float(option_price),
            "highest_price_usd": float(option_price),
            "highest_profit_pct": 0.0,
            "trailing_stop_price_usd": 0.0,
            "quantity": float(risk_decision.quantity),
            "capital_at_risk": float(risk_decision.risk_amount_usd),
            "current_value_usd": float(risk_decision.risk_amount_usd),
            "unrealized_pnl_usd": 0.0,
            "unrealized_pnl_pct": 0.0,
            "confidence_score": float(mc.mci),
            "dynamic_risk_pct": float(risk_decision.risk_pct),
            "dynamic_risk_reason": risk_decision.reason,
            "confidence_bucket": risk_decision.confidence_bucket,
            "drawdown_multiplier": risk_decision.drawdown_multiplier,
            "liquidity_multiplier": risk_decision.liquidity_multiplier,
            "portfolio_multiplier": risk_decision.portfolio_multiplier,
            "hybrid_exit_reason": "new_position",
            **exit_plan_columns(exit_plan),
        })
        cash -= risk_decision.risk_amount_usd
        current_open_risk_pct += risk_decision.risk_pct
        new_positions.append(position)

    if not new_positions:
        print("Candidates were selected, but no positions could be sized/opened.")
        return pd.DataFrame()

    updated_positions = pd.concat([open_positions, pd.DataFrame(new_positions)], ignore_index=True)
    save_open_positions(updated_positions, config)
    save_cash(cash, config)
    opened = pd.DataFrame(new_positions)
    print(f"Opened {len(opened)} new paper position(s).")
    print(f"Current paper cash: ${cash:,.2f}")
    print_new_positions_table(opened)
    return opened


def pct_string(value: Any) -> str:
    number = safe_float(value)
    return "" if number is None else f"{number:.2%}"


def num_string(value: Any, digits: int = 2) -> str:
    number = safe_float(value)
    return "" if number is None else f"{number:.{digits}f}"


def print_new_positions_table(opened: pd.DataFrame) -> None:
    if opened.empty:
        return
    cols = [c for c in ["instrument_name", "option_type", "strike", "days_to_expiry", "entry_price_usd", "quantity", "capital_at_risk", "mci", "confidence_bucket", "dynamic_risk_pct", "dynamic_soft_take_profit_pct", "dynamic_stop_loss_pct"] if c in opened.columns]
    if not cols:
        return
    display = opened[cols].copy().rename(columns={
        "instrument_name": "instrument",
        "option_type": "type",
        "days_to_expiry": "DTE",
        "entry_price_usd": "entry",
        "capital_at_risk": "risk_$",
        "dynamic_risk_pct": "risk_%",
        "dynamic_soft_take_profit_pct": "soft_TP",
        "dynamic_stop_loss_pct": "stop",
    })
    for column in ["risk_%", "soft_TP", "stop"]:
        if column in display.columns:
            display[column] = display[column].apply(pct_string)
    print("\n--- NEWLY OPENED POSITIONS ---")
    print(display.to_string(index=False))


def print_open_positions_table(config: PaperTraderConfig | None = None, max_rows: int = 50) -> None:
    if config is None:
        config = PaperTraderConfig()
    positions = load_open_positions(config)
    print("\n========== OPEN PAPER POSITIONS ==========")
    if positions.empty:
        print("No open paper positions.\n")
        return
    data = open_only(positions)
    if data.empty:
        print("No open paper positions.\n")
        return
    for col in ["strike", "days_to_expiry", "entry_price_usd", "current_price_usd", "quantity", "capital_at_risk", "current_value_usd", "unrealized_pnl_usd", "unrealized_pnl_pct", "mci", "dynamic_risk_pct"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    if "current_value_usd" not in data.columns:
        data["current_value_usd"] = numeric_series(data, "current_price_usd", 0.0) * numeric_series(data, "quantity", 0.0)
    if "unrealized_pnl_usd" not in data.columns:
        data["unrealized_pnl_usd"] = (numeric_series(data, "current_price_usd", 0.0) - numeric_series(data, "entry_price_usd", 0.0)) * numeric_series(data, "quantity", 0.0)
    if "unrealized_pnl_pct" not in data.columns:
        entry = numeric_series(data, "entry_price_usd", 1.0).replace(0, 1.0)
        data["unrealized_pnl_pct"] = numeric_series(data, "current_price_usd", 0.0) / entry - 1.0
    data["PnL_%"] = data["unrealized_pnl_pct"].apply(pct_string)
    if "dynamic_risk_pct" in data.columns:
        data["risk_%"] = data["dynamic_risk_pct"].apply(pct_string)
    cols = [c for c in ["instrument_name", "option_type", "strike", "days_to_expiry", "entry_price_usd", "current_price_usd", "quantity", "capital_at_risk", "current_value_usd", "unrealized_pnl_usd", "PnL_%", "mci", "confidence_bucket", "risk_%", "hybrid_exit_reason"] if c in data.columns]
    visible = data[cols].head(max_rows).copy().rename(columns={"instrument_name": "instrument", "option_type": "type", "days_to_expiry": "DTE", "entry_price_usd": "entry", "current_price_usd": "current", "capital_at_risk": "risk_$", "unrealized_pnl_usd": "PnL_$", "hybrid_exit_reason": "exit_state"})
    for col in visible.columns:
        if pd.api.types.is_numeric_dtype(visible[col]):
            visible[col] = visible[col].round(4)
    print(visible.to_string(index=False))
    print("------------------------------------------")
    print(f"Open positions: {len(data)}")
    print(f"Total open value: ${calculate_open_position_value(data):,.2f}")
    print(f"Total capital risk: ${calculate_open_risk_amount(data):,.2f}")
    print(f"Total unrealized PnL: ${calculate_unrealized_pnl_amount(data):,.2f}")
    print("==========================================")


def print_paper_account_summary(config: PaperTraderConfig | None = None) -> None:
    if config is None:
        config = PaperTraderConfig()
    cash = load_cash(config)
    positions = load_open_positions(config)
    open_value = calculate_open_position_value(positions)
    unrealized_pnl = calculate_unrealized_pnl_amount(positions)
    open_risk = calculate_open_risk_amount(positions)
    total_equity = cash + open_value
    total_return = total_equity / config.initial_cash - 1.0 if config.initial_cash > 0 else 0.0
    invested_pct = open_risk / config.initial_cash if config.initial_cash > 0 else 0.0
    print("\n========== PAPER ACCOUNT SUMMARY ==========")
    print(f"Starting equity: ${config.initial_cash:,.2f}")
    print(f"Free cash: ${cash:,.2f}")
    print(f"Open position value: ${open_value:,.2f}")
    print(f"Total equity: ${total_equity:,.2f}")
    print(f"Unrealized PnL: ${unrealized_pnl:,.2f}")
    print(f"Total return: {total_return:.2%}")
    print("-------------------------------------------")
    print(f"Open positions: {len(positions)}")
    print(f"Target positions: {config.target_positions}")
    print(f"Max positions: {config.max_positions}")
    print("-------------------------------------------")
    print(f"Capital at risk: ${open_risk:,.2f}")
    print(f"Capital invested %: {invested_pct:.2%}")
    print(f"Max open risk cap: ${config.initial_cash * config.max_total_open_risk_pct:,.2f}")
    print(f"Min risk/trade: {config.min_risk_per_trade:.2%}")
    print(f"Normal max risk/trade: {config.normal_max_risk_per_trade:.2%}")
    print(f"Exceptional max risk/trade: {config.exceptional_max_risk_per_trade:.2%}")
    print("==========================================")


if __name__ == "__main__":
    open_paper_trades(PaperTraderConfig())
