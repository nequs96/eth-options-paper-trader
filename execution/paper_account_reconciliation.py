"""
execution/paper_account_reconciliation.py

Paper account reconciliation.

Checks whether cash, open positions, and trade history are internally consistent.
This should be used as a safety gate before opening new paper trades.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from execution.paper_trader import PaperTraderConfig

RECONCILIATION_FILE = "outputs/paper_account_reconciliation.csv"
TOLERANCE = 0.01


def load_csv_if_exists(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists() or file_path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def numeric_column(df: pd.DataFrame, possible_columns: list[str], default: float = 0.0) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    for column in possible_columns:
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce").fillna(default)
    return pd.Series([default] * len(df), index=df.index, dtype=float)


def load_cash(config: PaperTraderConfig) -> float:
    cash_df = load_csv_if_exists(config.initial_cash_file)
    if cash_df.empty:
        return float(config.initial_cash)
    for column in ["cash", "paper_cash", "current_cash"]:
        if column in cash_df.columns:
            value = pd.to_numeric(cash_df[column], errors="coerce").dropna()
            if not value.empty:
                return float(value.iloc[-1])
    return float(config.initial_cash)


def calculate_open_current_value(open_positions: pd.DataFrame) -> float:
    if open_positions.empty:
        return 0.0
    if "status" in open_positions.columns:
        open_positions = open_positions[open_positions["status"].astype(str).str.lower().eq("open")]
    values = numeric_column(open_positions, ["current_value_usd", "market_value_usd"], 0.0)
    if values.empty or float(values.sum()) == 0.0:
        prices = numeric_column(open_positions, ["current_price_usd", "entry_price_usd"], 0.0)
        quantities = numeric_column(open_positions, ["quantity"], 0.0)
        return float((prices * quantities).sum())
    return float(values.sum())


def calculate_unrealized_pnl(open_positions: pd.DataFrame) -> float:
    if open_positions.empty:
        return 0.0
    if "status" in open_positions.columns:
        open_positions = open_positions[open_positions["status"].astype(str).str.lower().eq("open")]
    pnl = numeric_column(open_positions, ["unrealized_pnl_usd", "unrealized_pnl"], 0.0)
    return float(pnl.sum()) if not pnl.empty else 0.0


def calculate_realized_pnl(trade_history: pd.DataFrame) -> float:
    if trade_history.empty:
        return 0.0
    closed = trade_history
    if "status" in closed.columns:
        closed = closed[closed["status"].astype(str).str.lower().eq("closed")]
    pnl = numeric_column(closed, ["pnl_usd", "pnl"], 0.0)
    return float(pnl.sum()) if not pnl.empty else 0.0


def calculate_expected_cash_from_history(config: PaperTraderConfig, trade_history: pd.DataFrame, open_positions: pd.DataFrame) -> float:
    expected_cash = float(config.initial_cash)

    if not trade_history.empty:
        statuses = trade_history.get("status", pd.Series([""] * len(trade_history))).astype(str).str.lower()
        opens = trade_history[statuses.eq("open")]
        closes = trade_history[statuses.eq("closed")]

        open_costs = numeric_column(opens, ["capital_at_risk", "entry_cost_usd", "cost_usd"], 0.0)
        close_values = numeric_column(closes, ["current_value_usd", "exit_value_usd", "proceeds_usd"], 0.0)

        expected_cash -= float(open_costs.sum()) if not open_costs.empty else 0.0
        expected_cash += float(close_values.sum()) if not close_values.empty else 0.0

    # If open trades are not logged as open rows in history, subtract current open capital.
    if not open_positions.empty:
        costs = numeric_column(open_positions, ["capital_at_risk", "entry_cost_usd"], 0.0)
        # Avoid double subtracting if history already contains open rows.
        if trade_history.empty or "status" not in trade_history.columns or not trade_history["status"].astype(str).str.lower().eq("open").any():
            expected_cash -= float(costs.sum()) if not costs.empty else 0.0

    return float(expected_cash)


def generate_reconciliation_report(
    config: PaperTraderConfig | None = None,
    output_file: str = RECONCILIATION_FILE,
) -> pd.DataFrame:
    if config is None:
        config = PaperTraderConfig()

    trade_history = load_csv_if_exists(config.trade_history_file)
    open_positions = load_csv_if_exists(config.positions_file)
    actual_cash = load_cash(config)

    expected_cash = calculate_expected_cash_from_history(config, trade_history, open_positions)
    cash_difference = actual_cash - expected_cash
    open_current_value = calculate_open_current_value(open_positions)
    realized_pnl = calculate_realized_pnl(trade_history)
    unrealized_pnl = calculate_unrealized_pnl(open_positions)
    equity_estimate = actual_cash + open_current_value
    reconciliation_ok = abs(cash_difference) <= TOLERANCE

    report = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp.utcnow().isoformat(),
                "actual_cash": float(actual_cash),
                "expected_cash": float(expected_cash),
                "cash_difference": float(cash_difference),
                "open_current_value": float(open_current_value),
                "realized_pnl": float(realized_pnl),
                "unrealized_pnl": float(unrealized_pnl),
                "equity_estimate": float(equity_estimate),
                "open_positions": int(len(open_positions)) if not open_positions.empty else 0,
                "reconciliation_ok": bool(reconciliation_ok),
            }
        ]
    )

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_file, index=False)
    print(f"Reconciliation OK: {reconciliation_ok}")
    return report


if __name__ == "__main__":
    generate_reconciliation_report()
