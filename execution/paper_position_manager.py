"""Manage open paper option positions with per-position dynamic exits."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from data.options_data import DeribitConfig, fetch_deribit_ticker
from models.trend_regime_model import get_trend_regime
from execution.dynamic_exit_config import exit_plan_from_position
from execution.hybrid_exit_rules import evaluate_hybrid_exit
from execution.paper_trader import PaperTraderConfig, append_trade_history, load_cash, load_open_positions, save_cash, save_open_positions


@dataclass
class LiveOptionQuote:
    instrument_name: str
    underlying_price_usd: float
    mark_price_usd: float
    bid_price_usd: float | None
    ask_price_usd: float | None
    selected_exit_price_usd: float


def safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if result != result else float(result)


def safe_timestamp(value: Any) -> pd.Timestamp | None:
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def extract_ticker_result(ticker: dict[str, Any]) -> dict[str, Any]:
    if isinstance(ticker, dict) and isinstance(ticker.get("result"), dict):
        return ticker["result"]
    if isinstance(ticker, dict):
        return ticker
    return {}


def option_price_to_usd_if_needed(price: Any, underlying_price_usd: float) -> float | None:
    value = safe_float(price)
    if value is None or value <= 0:
        return None
    return float(value * underlying_price_usd) if value < 10 else float(value)


def get_live_option_quote_usd(instrument_name: str, fallback_underlying_price: float, deribit_config: DeribitConfig) -> LiveOptionQuote | None:
    try:
        ticker = fetch_deribit_ticker(instrument_name, deribit_config)
    except Exception as error:
        print(f"Could not fetch ticker for {instrument_name}: {error}")
        return None
    data = extract_ticker_result(ticker)
    underlying = safe_float(data.get("underlying_price")) or safe_float(data.get("index_price")) or safe_float(data.get("estimated_delivery_price")) or safe_float(fallback_underlying_price)
    if underlying is None or underlying <= 0:
        return None
    mark = option_price_to_usd_if_needed(data.get("mark_price"), underlying)
    last = option_price_to_usd_if_needed(data.get("last_price"), underlying)
    bid = option_price_to_usd_if_needed(data.get("best_bid_price"), underlying)
    ask = option_price_to_usd_if_needed(data.get("best_ask_price"), underlying)
    mark_or_last = mark or last
    if mark_or_last is None or mark_or_last <= 0:
        return None
    selected_exit = bid if bid is not None and bid > 0 else mark_or_last
    return LiveOptionQuote(instrument_name, float(underlying), float(mark_or_last), bid, ask, float(selected_exit))


def calculate_elapsed_days(position: pd.Series, now: pd.Timestamp) -> float:
    updated_at = safe_timestamp(position.get("updated_at")) or safe_timestamp(position.get("opened_at"))
    if updated_at is None:
        return 0.0
    return max((now - updated_at).total_seconds() / 86400.0, 0.0)


def update_days_to_expiry(current_days_to_expiry: float, elapsed_days: float) -> float:
    return max(float(current_days_to_expiry) - float(elapsed_days), 0.0)


def manage_paper_positions(trader_config: PaperTraderConfig | None = None, take_profit_pct: float | None = None, stop_loss_pct: float | None = None, min_days_to_expiry: float | None = None) -> None:
    if trader_config is None:
        trader_config = PaperTraderConfig()
    positions = load_open_positions(trader_config)
    if positions.empty:
        print("No open paper positions.")
        return

    cash = load_cash(trader_config)
    now = pd.Timestamp.utcnow()
    deribit_config = DeribitConfig()
    try:
        trend_regime = get_trend_regime()
    except Exception as error:
        print(f"WARNING: Could not load trend regime for hybrid exits: {error}")
        trend_regime = None

    remaining_rows: list[dict[str, Any]] = []
    closed_rows: list[dict[str, Any]] = []

    for _, position in positions.iterrows():
        instrument = str(position.get("instrument_name", "")).strip()
        if not instrument:
            continue
        entry = safe_float(position.get("entry_price_usd"))
        qty = safe_float(position.get("quantity"))
        dte = safe_float(position.get("days_to_expiry"))
        fallback = safe_float(position.get("spot_price")) or safe_float(position.get("underlying_price_usd")) or 0.0
        if entry is None or entry <= 0 or qty is None or qty <= 0 or dte is None:
            row = position.to_dict()
            row.update({"status": "invalid_position_data", "updated_at": now.isoformat()})
            remaining_rows.append(row)
            continue

        updated_dte = update_days_to_expiry(dte, calculate_elapsed_days(position, now))
        quote = get_live_option_quote_usd(instrument, fallback, deribit_config)
        if quote is None:
            row = position.to_dict()
            row.update({"updated_at": now.isoformat(), "days_to_expiry": float(updated_dte), "last_update_status": "quote_unavailable", "status": "open"})
            remaining_rows.append(row)
            continue

        current = float(quote.selected_exit_price_usd)
        previous_high = safe_float(position.get("highest_price_usd")) or entry
        exit_config = exit_plan_from_position(position)
        decision = evaluate_hybrid_exit(str(position.get("option_type", "")), entry, current, previous_high, updated_dte, trend_regime, exit_config)

        pnl_usd = (current - entry) * qty
        pnl_pct = current / entry - 1.0
        current_value = current * qty

        if decision.should_close:
            cash += current_value
            closed = position.to_dict()
            closed.update({
                "status": "closed",
                "closed_at": now.isoformat(),
                "updated_at": now.isoformat(),
                "exit_price_usd": float(current),
                "exit_value_usd": float(current_value),
                "proceeds_usd": float(current_value),
                "exit_mark_price_usd": float(quote.mark_price_usd),
                "exit_bid_price_usd": quote.bid_price_usd,
                "exit_ask_price_usd": quote.ask_price_usd,
                "exit_underlying_price_usd": float(quote.underlying_price_usd),
                "days_to_expiry": float(updated_dte),
                "pnl_usd": float(pnl_usd),
                "pnl_pct": float(pnl_pct),
                "close_reason": decision.reason,
                "hybrid_exit_reason": decision.reason,
                "highest_price_usd": float(decision.highest_price_usd),
                "highest_profit_pct": float(decision.highest_profit_pct),
                "trailing_stop_price_usd": float(decision.trailing_stop_price_usd),
                "trend_supportive_at_exit": bool(decision.trend_supportive),
                "regime_hostile_at_exit": bool(decision.regime_hostile),
                "trend_strength_at_exit": float(decision.trend_strength),
                "volatility_contracting_at_exit": bool(decision.volatility_contracting),
            })
            closed_rows.append(closed)
            print(f"Closed {instrument}: reason={decision.reason}, PnL={pnl_pct:.2%}")
        else:
            row = position.to_dict()
            row.update({
                "updated_at": now.isoformat(),
                "spot_price": float(quote.underlying_price_usd),
                "underlying_price_usd": float(quote.underlying_price_usd),
                "days_to_expiry": float(updated_dte),
                "current_price_usd": float(current),
                "mark_price_usd": float(quote.mark_price_usd),
                "bid_price_usd": quote.bid_price_usd,
                "ask_price_usd": quote.ask_price_usd,
                "current_value_usd": float(current_value),
                "unrealized_pnl_usd": float(pnl_usd),
                "unrealized_pnl_pct": float(pnl_pct),
                "highest_price_usd": float(decision.highest_price_usd),
                "highest_profit_pct": float(decision.highest_profit_pct),
                "trailing_stop_price_usd": float(decision.trailing_stop_price_usd),
                "hybrid_exit_reason": decision.reason,
                "trend_supportive": bool(decision.trend_supportive),
                "regime_hostile": bool(decision.regime_hostile),
                "trend_strength": float(decision.trend_strength),
                "volatility_contracting": bool(decision.volatility_contracting),
                "last_update_status": "updated",
                "status": "open",
            })
            remaining_rows.append(row)

    save_open_positions(pd.DataFrame(remaining_rows), trader_config)
    save_cash(cash, trader_config)
    if closed_rows:
        append_trade_history(pd.DataFrame(closed_rows), trader_config)
    print(f"Position management done. Open={len(remaining_rows)}, Closed={len(closed_rows)}, Cash=${cash:,.2f}")


if __name__ == "__main__":
    manage_paper_positions(PaperTraderConfig())
