"""
execution/paper_performance_report.py

Generates a lightweight paper trading performance report.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from execution.paper_trader import PaperTraderConfig
from execution.paper_account_reconciliation import load_csv_if_exists, load_cash, calculate_open_current_value, calculate_unrealized_pnl

SUMMARY_FILE = "outputs/paper_performance_summary.csv"


def numeric_column(df: pd.DataFrame, possible_columns: list[str], default: float = 0.0) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    for column in possible_columns:
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce").fillna(default)
    return pd.Series([default] * len(df), index=df.index, dtype=float)


def calculate_closed_trade_metrics(trade_history: pd.DataFrame) -> dict[str, float | int]:
    if trade_history.empty:
        return {
            "closed_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "realized_pnl": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "profit_factor": 0.0,
        }

    closed = trade_history
    if "status" in closed.columns:
        closed = closed[closed["status"].astype(str).str.lower().eq("closed")]

    pnl = numeric_column(closed, ["pnl_usd", "pnl"], 0.0)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_profit = float(wins.sum()) if not wins.empty else 0.0
    gross_loss = abs(float(losses.sum())) if not losses.empty else 0.0

    closed_count = int(len(pnl))
    return {
        "closed_trades": closed_count,
        "winning_trades": int(len(wins)),
        "losing_trades": int(len(losses)),
        "win_rate": float(len(wins) / closed_count) if closed_count else 0.0,
        "realized_pnl": float(pnl.sum()) if not pnl.empty else 0.0,
        "average_win": float(wins.mean()) if not wins.empty else 0.0,
        "average_loss": float(losses.mean()) if not losses.empty else 0.0,
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else 0.0,
    }


def generate_paper_performance_report(
    config: PaperTraderConfig | None = None,
    output_file: str = SUMMARY_FILE,
) -> pd.DataFrame:
    if config is None:
        config = PaperTraderConfig()

    trade_history = load_csv_if_exists(config.trade_history_file)
    open_positions = load_csv_if_exists(config.positions_file)
    cash = load_cash(config)

    closed_metrics = calculate_closed_trade_metrics(trade_history)
    open_value = calculate_open_current_value(open_positions)
    unrealized_pnl = calculate_unrealized_pnl(open_positions)
    equity = cash + open_value
    total_return = equity / config.initial_cash - 1.0 if config.initial_cash > 0 else 0.0

    summary = {
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "cash": float(cash),
        "open_positions": int(len(open_positions)) if not open_positions.empty else 0,
        "open_current_value": float(open_value),
        "unrealized_pnl": float(unrealized_pnl),
        "estimated_equity": float(equity),
        "total_return": float(total_return),
        **closed_metrics,
    }

    result = pd.DataFrame([summary])
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_file, index=False)
    print("Paper performance report generated.")
    return result


if __name__ == "__main__":
    generate_paper_performance_report()
