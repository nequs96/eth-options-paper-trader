"""
execution/paper_position_manager.py

Manages open paper option positions using live Deribit prices.

This module:
- loads open paper positions
- fetches live Deribit option prices
- calculates unrealized PnL
- closes positions based on exit rules
- updates paper cash & trade history

Important:
This is paper trading only.
No real orders are placed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from data.options_data import fetch_deribit_ticker, DeribitConfig
from execution.paper_trader import (
    PaperTraderConfig,
    load_open_positions,
    save_open_positions,
    append_trade_history,
    load_cash,
    save_cash,
)


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(f):
        return None
    return float(f)


def get_live_option_price_usd(
    instrument_name: str,
    underlying_price: float,
    deribit_config: DeribitConfig,
) -> float | None:
    """
    Fetch live option price from Deribit ticker.
    Uses mid bid/ask → mark → last.
    """

    ticker = fetch_deribit_ticker(
        instrument_name=instrument_name,
        config=deribit_config,
    )

    bid = safe_float(ticker.get("best_bid_price"))
    ask = safe_float(ticker.get("best_ask_price"))
    mark = safe_float(ticker.get("mark_price"))
    last = safe_float(ticker.get("last_price"))

    # Prices are quoted in ETH → convert to USD
    def to_usd(p: float | None) -> float | None:
        if p is None or underlying_price <= 0:
            return None
        return p * underlying_price

    bid_usd = to_usd(bid)
    ask_usd = to_usd(ask)

    if bid_usd and ask_usd and ask_usd >= bid_usd:
        return (bid_usd + ask_usd) / 2.0

    mark_usd = to_usd(mark)
    if mark_usd and mark_usd > 0:
        return mark_usd

    last_usd = to_usd(last)
    if last_usd and last_usd > 0:
        return last_usd

    return None


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
    """

    if entry_price <= 0:
        return True, "invalid_entry_price"

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
) -> None:
    """
    Update and manage open paper positions.
    """

    if trader_config is None:
        trader_config = PaperTraderConfig()

    open_positions = load_open_positions(trader_config)

    if open_positions.empty:
        print("No open paper positions.")
        return

    cash = load_cash(trader_config)

    deribit_config = DeribitConfig()

    closed_trades = []
    updated_positions = []

    print("Updating paper positions...")

    for _, pos in open_positions.iterrows():
        instrument = str(pos["instrument_name"])
        entry_price = float(pos["entry_price_usd"])
        quantity = float(pos["quantity"])
        days_to_expiry = float(pos["days_to_expiry"])
        underlying_price = float(pos["spot_price"])

        live_price = get_live_option_price_usd(
            instrument_name=instrument,
            underlying_price=underlying_price,
            deribit_config=deribit_config,
        )

        if live_price is None:
            updated_positions.append(pos)
            continue

        should_close, reason = should_close_position(
            entry_price=entry_price,
            current_price=live_price,
            days_to_expiry=days_to_expiry,
            take_profit_pct=take_profit_pct,
            stop_loss_pct=stop_loss_pct,
        )

        if should_close:
            pnl = (live_price - entry_price) * quantity
            cash += live_price * quantity

            closed_trade = pos.to_dict()
            closed_trade.update(
                {
                    "closed_at": str(pd.Timestamp.utcnow()),
                    "exit_price_usd": live_price,
                    "pnl": pnl,
                    "close_reason": reason,
                    "status": "closed",
                }
            )
            closed_trades.append(closed_trade)
        else:
            updated_pos = pos.to_dict()
            updated_pos["unrealized_pnl"] = (live_price - entry_price) * quantity
            updated_positions.append(updated_pos)

    # Save updates
    save_open_positions(pd.DataFrame(updated_positions), trader_config)
    append_trade_history(pd.DataFrame(closed_trades), trader_config)
    save_cash(cash, trader_config)

    print("========== PAPER POSITION UPDATE ==========")
    print(f"Closed positions: {len(closed_trades)}")
    print(f"Remaining open positions: {len(updated_positions)}")
    print(f"Updated cash balance: ${cash:,.2f}")
    print("===========================================")


if __name__ == "__main__":
    config = PaperTraderConfig()
    manage_paper_positions(config)