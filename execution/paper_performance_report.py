"""
execution/paper_performance_report.py

Clear paper trading performance report.

This module reads:
- outputs/paper_trade_history.csv
- outputs/paper_open_positions.csv
- outputs/paper_cash.csv

And creates:
- outputs/paper_performance_summary.csv

It shows:
- closed trades
- wins / losses
- realized PnL
- unrealized PnL
- current cash
- estimated equity
- win rate
- average win/loss
- profit factor
- basic cash reconciliation
"""

from pathlib import Path
import math

import pandas as pd


TRADE_HISTORY_FILE = "outputs/paper_trade_history.csv"
OPEN_POSITIONS_FILE = "outputs/paper_open_positions.csv"
CASH_FILE = "outputs/paper_cash.csv"
SUMMARY_FILE = "outputs/paper_performance_summary.csv"

INITIAL_CASH = 10_000.0


def load_csv_if_exists(file_path: str) -> pd.DataFrame:
    path = Path(file_path)

    if not path.exists():
        return pd.DataFrame()

    if path.stat().st_size == 0:
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def load_cash(file_path: str = CASH_FILE) -> float:
    data = load_csv_if_exists(file_path)

    if data.empty or "cash" not in data.columns:
        return INITIAL_CASH

    cash = pd.to_numeric(data["cash"], errors="coerce").dropna()

    if cash.empty:
        return INITIAL_CASH

    return float(cash.iloc[0])


def get_numeric_column(
    data: pd.DataFrame,
    possible_columns: list[str],
    default: float = 0.0,
) -> pd.Series:
    """
    Return first available numeric column from possible column names.
    """

    for column in possible_columns:
        if column in data.columns:
            return pd.to_numeric(data[column], errors="coerce").fillna(default)

    return pd.Series([default] * len(data), index=data.index)


def calculate_closed_trade_metrics(trade_history: pd.DataFrame) -> dict:
    """
    Calculate performance metrics from closed trades.
    Supports both old and new PnL column names:
    - pnl
    - pnl_usd
    """

    empty_metrics = {
        "closed_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "total_realized_pnl": 0.0,
        "win_rate": 0.0,
        "average_win": 0.0,
        "average_loss": 0.0,
        "profit_factor": 0.0,
        "best_trade": 0.0,
        "worst_trade": 0.0,
    }

    if trade_history.empty:
        return empty_metrics

    data = trade_history.copy()

    if "status" in data.columns:
        data = data[data["status"].astype(str).str.lower() == "closed"].copy()

    if data.empty:
        return empty_metrics

    data["pnl_normalized"] = get_numeric_column(data, ["pnl_usd", "pnl"])

    data = data.dropna(subset=["pnl_normalized"])

    if data.empty:
        return empty_metrics

    wins = data[data["pnl_normalized"] > 0]["pnl_normalized"]
    losses = data[data["pnl_normalized"] < 0]["pnl_normalized"]

    closed_trades = len(data)
    winning_trades = len(wins)
    losing_trades = len(losses)

    total_realized_pnl = float(data["pnl_normalized"].sum())
    win_rate = winning_trades / closed_trades if closed_trades > 0 else 0.0

    average_win = float(wins.mean()) if not wins.empty else 0.0
    average_loss = float(losses.mean()) if not losses.empty else 0.0

    gross_profit = float(wins.sum()) if not wins.empty else 0.0
    gross_loss = abs(float(losses.sum())) if not losses.empty else 0.0

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = math.inf
    else:
        profit_factor = 0.0

    return {
        "closed_trades": int(closed_trades),
        "winning_trades": int(winning_trades),
        "losing_trades": int(losing_trades),
        "total_realized_pnl": total_realized_pnl,
        "win_rate": float(win_rate),
        "average_win": average_win,
        "average_loss": average_loss,
        "profit_factor": profit_factor,
        "best_trade": float(data["pnl_normalized"].max()),
        "worst_trade": float(data["pnl_normalized"].min()),
    }


def calculate_open_position_metrics(open_positions: pd.DataFrame) -> dict:
    """
    Calculate metrics for currently open positions.
    Supports both old and new unrealized PnL column names:
    - unrealized_pnl
    - unrealized_pnl_usd
    """

    if open_positions.empty:
        return {
            "open_positions": 0,
            "open_capital_at_risk": 0.0,
            "open_unrealized_pnl": 0.0,
            "open_current_value": 0.0,
        }

    data = open_positions.copy()

    capital_at_risk = get_numeric_column(data, ["capital_at_risk"])
    unrealized_pnl = get_numeric_column(data, ["unrealized_pnl_usd", "unrealized_pnl"])

    if "current_value_usd" in data.columns:
        current_value = pd.to_numeric(data["current_value_usd"], errors="coerce").fillna(0.0)
    else:
        current_value = capital_at_risk + unrealized_pnl

    return {
        "open_positions": int(len(data)),
        "open_capital_at_risk": float(capital_at_risk.sum()),
        "open_unrealized_pnl": float(unrealized_pnl.sum()),
        "open_current_value": float(current_value.sum()),
    }


def generate_paper_performance_report() -> pd.DataFrame:
    trade_history = load_csv_if_exists(TRADE_HISTORY_FILE)
    open_positions = load_csv_if_exists(OPEN_POSITIONS_FILE)
    cash = load_cash(CASH_FILE)

    closed_metrics = calculate_closed_trade_metrics(trade_history)
    open_metrics = calculate_open_position_metrics(open_positions)

    equity_estimate = cash + float(open_metrics["open_current_value"])

    total_pnl_estimate = equity_estimate - INITIAL_CASH

    summary = {
        **closed_metrics,
        **open_metrics,
        "initial_cash": INITIAL_CASH,
        "current_cash": cash,
        "account_equity_estimate": equity_estimate,
        "total_pnl_estimate": total_pnl_estimate,
        "total_return_pct": total_pnl_estimate / INITIAL_CASH if INITIAL_CASH > 0 else 0.0,
    }

    summary_df = pd.DataFrame([summary])

    Path(SUMMARY_FILE).parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(SUMMARY_FILE, index=False)

    print("========== PAPER PERFORMANCE REPORT ==========")
    print(f"Initial cash:           ${summary['initial_cash']:,.2f}")
    print(f"Current cash:           ${summary['current_cash']:,.2f}")
    print("----------------------------------------------")
    print(f"Closed trades:          {summary['closed_trades']}")
    print(f"Winning trades:         {summary['winning_trades']}")
    print(f"Losing trades:          {summary['losing_trades']}")
    print(f"Win rate:               {summary['win_rate']:.2%}")
    print("----------------------------------------------")
    print(f"Total realized PnL:     ${summary['total_realized_pnl']:,.2f}")
    print(f"Average win:            ${summary['average_win']:,.2f}")
    print(f"Average loss:           ${summary['average_loss']:,.2f}")
    print(f"Best trade:             ${summary['best_trade']:,.2f}")
    print(f"Worst trade:            ${summary['worst_trade']:,.2f}")
    print(f"Profit factor:          {summary['profit_factor']}")
    print("----------------------------------------------")
    print(f"Open positions:         {summary['open_positions']}")
    print(f"Open capital at risk:   ${summary['open_capital_at_risk']:,.2f}")
    print(f"Open current value:     ${summary['open_current_value']:,.2f}")
    print(f"Open unrealized PnL:    ${summary['open_unrealized_pnl']:,.2f}")
    print("----------------------------------------------")
    print(f"Equity estimate:        ${summary['account_equity_estimate']:,.2f}")
    print(f"Total PnL estimate:     ${summary['total_pnl_estimate']:,.2f}")
    print(f"Total return:           {summary['total_return_pct']:.2%}")
    print("----------------------------------------------")
    print(f"Saved summary to:       {SUMMARY_FILE}")
    print("==============================================")

    return summary_df


if __name__ == "__main__":
    generate_paper_performance_report()