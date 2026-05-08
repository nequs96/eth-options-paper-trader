"""
execution/paper_position_manager.py

Manages open paper option positions using live Deribit prices.

This module:
- loads open paper positions
- fetches live Deribit option prices
- marks positions to market
- calculates unrealized PnL
- closes positions based on exit rules
- updates paper cash
- appends closed trades to trade history

Important:
This is paper trading only.
No real orders are placed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from models.trend_regime_model import get_trend_regime
from execution.hybrid_exit_rules import HybridExitConfig, evaluate_hybrid_exit

from data.options_data import fetch_deribit_ticker, DeribitConfig
from execution.paper_trader import (
    PaperTraderConfig,
    load_open_positions,
    save_open_positions,
    append_trade_history,
    load_cash,
    save_cash,
)


@dataclass
class LiveOptionQuote:
    """
    Live option quote converted to USD.
    """

    price_usd: float
    underlying_price_usd: float
    source: str


def safe_float(value: Any) -> float | None:
    """
    Safely convert value to float.
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


def safe_timestamp(value: Any) -> pd.Timestamp | None:
    """
    Safely parse timestamp as UTC.
    """

    if value is None or pd.isna(value):
        return None

    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None

    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")

    return ts


def extract_ticker_result(ticker: dict[str, Any]) -> dict[str, Any]:
    """
    Some API wrappers return the ticker directly.
    Others return {'result': {...}}.

    This function supports both.
    """

    if not isinstance(ticker, dict):
        return {}

    result = ticker.get("result")

    if isinstance(result, dict):
        return result

    return ticker


def get_live_option_quote_usd(
    instrument_name: str,
    fallback_underlying_price: float,
    deribit_config: DeribitConfig,
) -> LiveOptionQuote | None:
    """
    Fetch live option quote from Deribit ticker and convert to USD.

    Deribit option prices are usually quoted in ETH.
    This function converts the option price into USD using the live underlying
    price from ticker if available.

    Price source priority:
    1. bid/ask midpoint
    2. mark price
    3. last price
    """

    try:
        raw_ticker = fetch_deribit_ticker(
            instrument_name=instrument_name,
            config=deribit_config,
        )
    except Exception as error:
        print(f"WARNING: Failed to fetch ticker for {instrument_name}: {error}")
        return None

    ticker = extract_ticker_result(raw_ticker)

    if not ticker:
        return None

    # Try to get live underlying price from ticker.
    underlying_price = (
        safe_float(ticker.get("underlying_price"))
        or safe_float(ticker.get("index_price"))
        or safe_float(ticker.get("estimated_delivery_price"))
        or safe_float(fallback_underlying_price)
    )

    if underlying_price is None or underlying_price <= 0:
        return None

    bid = safe_float(ticker.get("best_bid_price"))
    ask = safe_float(ticker.get("best_ask_price"))
    mark = safe_float(ticker.get("mark_price"))
    last = safe_float(ticker.get("last_price"))

    def to_usd(option_price_in_eth: float | None) -> float | None:
        if option_price_in_eth is None:
            return None
        if option_price_in_eth <= 0:
            return None
        return float(option_price_in_eth * underlying_price)

    bid_usd = to_usd(bid)
    ask_usd = to_usd(ask)

    if (
        bid_usd is not None
        and ask_usd is not None
        and bid_usd > 0
        and ask_usd > 0
        and ask_usd >= bid_usd
    ):
        return LiveOptionQuote(
            price_usd=float((bid_usd + ask_usd) / 2.0),
            underlying_price_usd=float(underlying_price),
            source="bid_ask_mid",
        )

    mark_usd = to_usd(mark)

    if mark_usd is not None and mark_usd > 0:
        return LiveOptionQuote(
            price_usd=float(mark_usd),
            underlying_price_usd=float(underlying_price),
            source="mark",
        )

    last_usd = to_usd(last)

    if last_usd is not None and last_usd > 0:
        return LiveOptionQuote(
            price_usd=float(last_usd),
            underlying_price_usd=float(underlying_price),
            source="last",
        )

    return None


def calculate_elapsed_days(position: pd.Series, now: pd.Timestamp) -> float:
    """
    Calculate elapsed days since the position was last updated.

    This prevents days_to_expiry from decreasing multiple times if you run
    the manager more than once in one day.
    """

    last_updated = None

    if "last_updated_at" in position.index:
        last_updated = safe_timestamp(position.get("last_updated_at"))

    if last_updated is None and "opened_at" in position.index:
        last_updated = safe_timestamp(position.get("opened_at"))

    if last_updated is None:
        return 0.0

    elapsed_seconds = (now - last_updated).total_seconds()

    if elapsed_seconds <= 0:
        return 0.0

    return float(elapsed_seconds / 86_400.0)


def update_days_to_expiry(
    current_days_to_expiry: float,
    elapsed_days: float,
) -> float:
    """
    Reduce days_to_expiry by elapsed days.
    """

    if current_days_to_expiry <= 0:
        return 0.0

    return float(max(current_days_to_expiry - elapsed_days, 0.0))


def should_close_position(
    entry_price: float,
    current_price: float,
    days_to_expiry: float,
    take_profit_pct: float,
    stop_loss_pct: float,
    min_days_to_expiry: float = 1.0,
) -> tuple[bool, str]:
    """
    Decide whether a paper position should be closed.

    For long options:
    - take_profit_pct = 0.40 means close at +40%
    - stop_loss_pct = -0.30 means close at -30%
    """

    if entry_price <= 0:
        return True, "invalid_entry_price"

    if current_price <= 0:
        return True, "invalid_current_price"

    pnl_pct = current_price / entry_price - 1.0

    if pnl_pct >= take_profit_pct:
        return True, "take_profit"

    if pnl_pct <= stop_loss_pct:
        return True, "stop_loss"

    if days_to_expiry <= min_days_to_expiry:
        return True, "near_expiry"

    return False, "hold"


def manage_paper_positions(
    trader_config: PaperTraderConfig | None = None,
    take_profit_pct: float = 0.40,
    stop_loss_pct: float = -0.30,
    min_days_to_expiry: float = 1.0,
) -> None:
    """
    Update and manage open paper positions.

    This function:
    - fetches live option prices
    - updates unrealized PnL
    - closes positions if exit rules trigger
    - updates cash after closes
    """

    if trader_config is None:
        trader_config = PaperTraderConfig()

    open_positions = load_open_positions(trader_config)

    if open_positions.empty:
        print("No open paper positions.")
        return

    cash = load_cash(trader_config)
    deribit_config = DeribitConfig()

    try:
        trend_regime = get_trend_regime()
    except Exception as error:
        print(f"WARNING: Could not load trend regime for hybrid exits: {error}")
        trend_regime = None

    hybrid_exit_config = HybridExitConfig(
        stop_loss_pct=stop_loss_pct,
        soft_take_profit_pct=take_profit_pct,
        hard_take_profit_pct=0.60,
        trailing_activation_pct=0.25,
        trailing_drawdown_pct=0.15,
        min_days_to_expiry=min_days_to_expiry,
        close_profitable_trade_on_trend_loss=True,
        close_profitable_trade_on_hostile_regime=True,
    )
    

    now = pd.Timestamp.now(tz="UTC")

    closed_trades: list[dict[str, Any]] = []
    updated_positions: list[dict[str, Any]] = []

    print("Updating paper positions...")

    for _, pos in open_positions.iterrows():
        instrument = str(pos.get("instrument_name", "")).strip()

        entry_price = safe_float(pos.get("entry_price_usd"))
        quantity = safe_float(pos.get("quantity"))
        original_days_to_expiry = safe_float(pos.get("days_to_expiry"))
        fallback_underlying_price = safe_float(pos.get("spot_price"))

        if not instrument:
            continue

        if entry_price is None or entry_price <= 0:
            continue

        if quantity is None or quantity <= 0:
            continue

        if original_days_to_expiry is None:
            original_days_to_expiry = 0.0

        if fallback_underlying_price is None or fallback_underlying_price <= 0:
            fallback_underlying_price = 0.0

        elapsed_days = calculate_elapsed_days(pos, now)

        updated_dte = update_days_to_expiry(
            current_days_to_expiry=float(original_days_to_expiry),
            elapsed_days=elapsed_days,
        )

        quote = get_live_option_quote_usd(
            instrument_name=instrument,
            fallback_underlying_price=fallback_underlying_price,
            deribit_config=deribit_config,
        )

        position_dict = pos.to_dict()

        if quote is None:
            # Keep position open but record that price update failed.
            position_dict["days_to_expiry"] = updated_dte
            position_dict["last_updated_at"] = str(now)
            position_dict["price_update_status"] = "failed"
            position_dict["status"] = "open"

            updated_positions.append(position_dict)
            continue

        current_price = float(quote.price_usd)
        current_value = current_price * quantity
        entry_value = entry_price * quantity

        unrealized_pnl = current_value - entry_value
        unrealized_pnl_pct = current_price / entry_price - 1.0

        previous_highest_price = safe_float(pos.get("highest_price_usd"))

        if previous_highest_price is None or previous_highest_price <= 0:
            previous_highest_price = entry_price

        exit_decision = evaluate_hybrid_exit(
            option_type=str(pos.get("option_type", "")),
            entry_price_usd=entry_price,
            current_price_usd=current_price,
            highest_price_usd=previous_highest_price,
            days_to_expiry=updated_dte,
            trend_regime=trend_regime,
            config=hybrid_exit_config,
        )

        should_close = exit_decision.should_close
        reason = exit_decision.reason

        if should_close:
            # Since cash was reduced when trade was opened,
            # closing a long option adds exit value back to cash.
            cash += current_value

            closed_trade = position_dict.copy()
            closed_trade.update(
                {
                    "closed_at": str(now),
                    "exit_price_usd": float(current_price),
                    "exit_value_usd": float(current_value),
                    "entry_value_usd": float(entry_value),
                    "pnl_usd": float(unrealized_pnl),
                    "pnl_pct": float(unrealized_pnl_pct),
                    "close_reason": reason,
                    "days_to_expiry_at_close": float(updated_dte),
                    "underlying_price_at_close": float(quote.underlying_price_usd),
                    "quote_source_at_close": quote.source,
                    "status": "closed",
                    "hybrid_exit_reason": reason,
                    "hybrid_exit_pnl_pct": float(exit_decision.pnl_pct),
                    "highest_price_usd": float(exit_decision.highest_price_usd),
                    "trailing_stop_price_usd": float(exit_decision.trailing_stop_price_usd),
                    "trend_supportive_at_exit": bool(exit_decision.trend_supportive),
                    "regime_hostile_at_exit": bool(exit_decision.regime_hostile),
                }
            )

            closed_trades.append(closed_trade)

        else:
            position_dict.update(
                {
                    "spot_price": float(quote.underlying_price_usd),
                    "days_to_expiry": float(updated_dte),
                    "current_price_usd": float(current_price),
                    "current_value_usd": float(current_value),
                    "entry_value_usd": float(entry_value),
                    "unrealized_pnl_usd": float(unrealized_pnl),
                    "unrealized_pnl_pct": float(unrealized_pnl_pct),
                    "quote_source": quote.source,
                    "last_updated_at": str(now),
                    "price_update_status": "ok",
                    "status": "open",
                    "highest_price_usd": float(exit_decision.highest_price_usd),
                    "trailing_stop_price_usd": float(exit_decision.trailing_stop_price_usd),
                    "hybrid_exit_pnl_pct": float(exit_decision.pnl_pct),
                    "trend_supportive": bool(exit_decision.trend_supportive),
                    "regime_hostile": bool(exit_decision.regime_hostile),
                }
            )

            updated_positions.append(position_dict)

    updated_positions_df = pd.DataFrame(updated_positions)
    closed_trades_df = pd.DataFrame(closed_trades)

    save_open_positions(updated_positions_df, trader_config)

    if not closed_trades_df.empty:
        append_trade_history(closed_trades_df, trader_config)

    save_cash(cash, trader_config)

    print("========== PAPER POSITION UPDATE ==========")
    print(f"Closed positions:          {len(closed_trades_df)}")
    print(f"Remaining open positions:  {len(updated_positions_df)}")
    print(f"Updated cash balance:      ${cash:,.2f}")

    if not updated_positions_df.empty and "unrealized_pnl_usd" in updated_positions_df.columns:
        unrealized = pd.to_numeric(
            updated_positions_df["unrealized_pnl_usd"],
            errors="coerce",
        ).fillna(0.0).sum()

        print(f"Open unrealized PnL:       ${float(unrealized):,.2f}")

    if not closed_trades_df.empty and "pnl_usd" in closed_trades_df.columns:
        realized = pd.to_numeric(
            closed_trades_df["pnl_usd"],
            errors="coerce",
        ).fillna(0.0).sum()

        print(f"Realized PnL this update:  ${float(realized):,.2f}")

    print("===========================================")


if __name__ == "__main__":
    config = PaperTraderConfig()
    manage_paper_positions(config)