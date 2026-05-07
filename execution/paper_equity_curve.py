"""
execution/paper_equity_curve.py

Generates paper trading equity curve over time.

Reads:
- outputs/paper_trade_history.csv
- outputs/paper_open_positions.csv
- outputs/paper_cash.csv

Creates:
- outputs/paper_equity_curve.csv
"""

from pathlib import Path
import pandas as pd


TRADE_HISTORY_FILE = "outputs/paper_trade_history.csv"
OPEN_POSITIONS_FILE = "outputs/paper_open_positions.csv"
CASH_FILE = "outputs/paper_cash.csv"
EQUITY_CURVE_FILE = "outputs/paper_equity_curve.csv"

INITIAL_CASH = 10_000.0


def load_csv(path: str) -> pd.DataFrame:
    p = Path(path)

    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()

    try:
        return pd.read_csv(p)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def load_cash() -> float:
    df = load_csv(CASH_FILE)

    if df.empty or "cash" not in df.columns:
        return INITIAL_CASH

    cash = pd.to_numeric(df["cash"], errors="coerce").dropna()

    if cash.empty:
        return INITIAL_CASH

    return float(cash.iloc[0])


def numeric_column(
    df: pd.DataFrame,
    possible_columns: list[str],
    default: float = 0.0,
) -> pd.Series:
    for col in possible_columns:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(default)

    return pd.Series([default] * len(df), index=df.index)


def calculate_open_current_value(positions: pd.DataFrame) -> float:
    if positions.empty:
        return 0.0

    if "current_value_usd" in positions.columns:
        return float(
            pd.to_numeric(
                positions["current_value_usd"],
                errors="coerce",
            ).fillna(0.0).sum()
        )

    capital_at_risk = numeric_column(positions, ["capital_at_risk"])
    unrealized = numeric_column(positions, ["unrealized_pnl_usd", "unrealized_pnl"])

    return float((capital_at_risk + unrealized).sum())


def generate_equity_curve() -> pd.DataFrame:
    trades = load_csv(TRADE_HISTORY_FILE)
    positions = load_csv(OPEN_POSITIONS_FILE)
    cash = load_cash()

    rows = []

    now = pd.Timestamp.now(tz="UTC")

    rows.append(
        {
            "timestamp": now,
            "equity": INITIAL_CASH,
            "cash": INITIAL_CASH,
            "open_current_value": 0.0,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "type": "initial",
        }
    )

    realized_pnl = 0.0

    if not trades.empty:
        closed = trades.copy()

        if "status" in closed.columns:
            closed = closed[closed["status"].astype(str).str.lower() == "closed"].copy()

        if not closed.empty:
            if "closed_at" in closed.columns:
                closed["timestamp"] = pd.to_datetime(
                    closed["closed_at"],
                    errors="coerce",
                    utc=True,
                )
            else:
                closed["timestamp"] = now

            closed["pnl_normalized"] = numeric_column(closed, ["pnl_usd", "pnl"])
            closed = closed.dropna(subset=["timestamp"])
            closed = closed.sort_values("timestamp")

            cumulative = 0.0

            for _, trade in closed.iterrows():
                pnl = float(trade["pnl_normalized"])
                cumulative += pnl

                rows.append(
                    {
                        "timestamp": trade["timestamp"],
                        "equity": INITIAL_CASH + cumulative,
                        "cash": None,
                        "open_current_value": None,
                        "realized_pnl": cumulative,
                        "unrealized_pnl": None,
                        "type": "closed_trade_estimate",
                    }
                )

            realized_pnl = cumulative

    open_current_value = calculate_open_current_value(positions)

    unrealized_pnl = 0.0
    if not positions.empty:
        unrealized_pnl = float(
            numeric_column(
                positions,
                ["unrealized_pnl_usd", "unrealized_pnl"],
            ).sum()
        )

    current_equity = cash + open_current_value

    rows.append(
        {
            "timestamp": now,
            "equity": current_equity,
            "cash": cash,
            "open_current_value": open_current_value,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "type": "current",
        }
    )

    equity_df = pd.DataFrame(rows)
    equity_df = equity_df.sort_values("timestamp").reset_index(drop=True)

    Path(EQUITY_CURVE_FILE).parent.mkdir(parents=True, exist_ok=True)
    equity_df.to_csv(EQUITY_CURVE_FILE, index=False)

    print("========== EQUITY CURVE ==========")
    print(f"Current cash:       ${cash:,.2f}")
    print(f"Open value:         ${open_current_value:,.2f}")
    print(f"Current equity:     ${current_equity:,.2f}")
    print(f"Saved to:           {EQUITY_CURVE_FILE}")
    print("=================================")

    return equity_df


if __name__ == "__main__":
    generate_equity_curve()