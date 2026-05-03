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
- wins
- losses
- total PnL
- win rate
- average win
- average loss
- profit factor
- current cash
- open unrealized PnL
"""

from pathlib import Path

import pandas as pd


TRADE_HISTORY_FILE = "outputs/paper_trade_history.csv"
OPEN_POSITIONS_FILE = "outputs/paper_open_positions.csv"
CASH_FILE = "outputs/paper_cash.csv"
SUMMARY_FILE = "outputs/paper_performance_summary.csv"


def load_csv_if_exists(file_path: str) -> pd.DataFrame:
    """
    Load CSV if it exists, otherwise return empty DataFrame.
    """

    path = Path(file_path)

    if not path.exists():
        return pd.DataFrame()

    data = pd.read_csv(path)

    return data


def load_cash(file_path: str = CASH_FILE) -> float:
    """
    Load paper cash balance.
    """

    data = load_csv_if_exists(file_path)

    if data.empty or "cash" not in data.columns:
        return 0.0

    cash = pd.to_numeric(data["cash"], errors="coerce").dropna()

    if cash.empty:
        return 0.0

    return float(cash.iloc[0])


def calculate_closed_trade_metrics(trade_history: pd.DataFrame) -> dict[str, float | int]:
    """
    Calculate performance metrics from closed trades.
    """

    if trade_history.empty or "pnl" not in trade_history.columns:
        return {
            "closed_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl": 0.0,
            "win_rate": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "profit_factor": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
        }

    data = trade_history.copy()

    if "status" in data.columns:
        data = data[data["status"] == "closed"].copy()

    data["pnl"] = pd.to_numeric(data["pnl"], errors="coerce")
    data = data.dropna(subset=["pnl"])

    if data.empty:
        return {
            "closed_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl": 0.0,
            "win_rate": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "profit_factor": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
        }

    wins = data[data["pnl"] > 0]["pnl"]
    losses = data[data["pnl"] < 0]["pnl"]

    closed_trades = len(data)
    winning_trades = len(wins)
    losing_trades = len(losses)

    total_pnl = float(data["pnl"].sum())
    win_rate = winning_trades / closed_trades if closed_trades > 0 else 0.0

    average_win = float(wins.mean()) if not wins.empty else 0.0
    average_loss = float(losses.mean()) if not losses.empty else 0.0

    gross_profit = float(wins.sum()) if not wins.empty else 0.0
    gross_loss = abs(float(losses.sum())) if not losses.empty else 0.0

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    best_trade = float(data["pnl"].max())
    worst_trade = float(data["pnl"].min())

    return {
        "closed_trades": closed_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "total_pnl": total_pnl,
        "win_rate": win_rate,
        "average_win": average_win,
        "average_loss": average_loss,
        "profit_factor": profit_factor,
        "best_trade": best_trade,
        "worst_trade": worst_trade,
    }


def calculate_open_position_metrics(open_positions: pd.DataFrame) -> dict[str, float | int]:
    """
    Calculate metrics for currently open positions.
    """

    if open_positions.empty:
        return {
            "open_positions": 0,
            "open_capital_at_risk": 0.0,
            "open_unrealized_pnl": 0.0,
        }

    data = open_positions.copy()

    if "capital_at_risk" in data.columns:
        data["capital_at_risk"] = pd.to_numeric(
            data["capital_at_risk"],
            errors="coerce",
        )
        open_capital_at_risk = float(data["capital_at_risk"].fillna(0.0).sum())
    else:
        open_capital_at_risk = 0.0

    if "unrealized_pnl" in data.columns:
        data["unrealized_pnl"] = pd.to_numeric(
            data["unrealized_pnl"],
            errors="coerce",
        )
        open_unrealized_pnl = float(data["unrealized_pnl"].fillna(0.0).sum())
    else:
        open_unrealized_pnl = 0.0

    return {
        "open_positions": len(data),
        "open_capital_at_risk": open_capital_at_risk,
        "open_unrealized_pnl": open_unrealized_pnl,
    }


def generate_paper_performance_report() -> pd.DataFrame:
    """
    Generate and save paper performance summary.
    """

    trade_history = load_csv_if_exists(TRADE_HISTORY_FILE)
    open_positions = load_csv_if_exists(OPEN_POSITIONS_FILE)
    cash = load_cash(CASH_FILE)

    closed_metrics = calculate_closed_trade_metrics(trade_history)
    open_metrics = calculate_open_position_metrics(open_positions)

    summary = {
        **closed_metrics,
        **open_metrics,
        "current_cash": cash,
        "account_equity_estimate": cash + float(open_metrics["open_capital_at_risk"]) + float(open_metrics["open_unrealized_pnl"]),
    }

    summary_df = pd.DataFrame([summary])

    Path(SUMMARY_FILE).parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(SUMMARY_FILE, index=False)

    print("========== PAPER PERFORMANCE REPORT ==========")
    print(f"Closed trades:          {summary['closed_trades']}")
    print(f"Winning trades:         {summary['winning_trades']}")
    print(f"Losing trades:          {summary['losing_trades']}")
    print(f"Win rate:               {summary['win_rate']:.2%}")
    print("----------------------------------------------")
    print(f"Total realized PnL:     ${summary['total_pnl']:,.2f}")
    print(f"Average win:            ${summary['average_win']:,.2f}")
    print(f"Average loss:           ${summary['average_loss']:,.2f}")
    print(f"Best trade:             ${summary['best_trade']:,.2f}")
    print(f"Worst trade:            ${summary['worst_trade']:,.2f}")
    print(f"Profit factor:          {summary['profit_factor']}")
    print("----------------------------------------------")
    print(f"Open positions:         {summary['open_positions']}")
    print(f"Open capital at risk:   ${summary['open_capital_at_risk']:,.2f}")
    print(f"Open unrealized PnL:    ${summary['open_unrealized_pnl']:,.2f}")
    print("----------------------------------------------")
    print(f"Current cash:           ${summary['current_cash']:,.2f}")
    print(f"Equity estimate:        ${summary['account_equity_estimate']:,.2f}")
    print("----------------------------------------------")
    print(f"Saved summary to:       {SUMMARY_FILE}")
    print("==============================================")

    return summary_df


if __name__ == "__main__":
    generate_paper_performance_report()