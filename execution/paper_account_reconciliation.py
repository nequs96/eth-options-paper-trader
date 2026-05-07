"""
execution/paper_account_reconciliation.py

Reconciles paper account state.

Checks:
- cash file
- open positions
- trade history
- expected cash from opens/closes
- equity estimate

Goal:
Prevent fake or unexplained paper profits.
"""

from pathlib import Path
import pandas as pd


TRADE_HISTORY_FILE = "outputs/paper_trade_history.csv"
OPEN_POSITIONS_FILE = "outputs/paper_open_positions.csv"
CASH_FILE = "outputs/paper_cash.csv"
RECONCILIATION_FILE = "outputs/paper_account_reconciliation.csv"

INITIAL_CASH = 10_000.0
TOLERANCE = 0.01


def load_csv_if_exists(path: str) -> pd.DataFrame:
    file_path = Path(path)

    if not file_path.exists() or file_path.stat().st_size == 0:
        return pd.DataFrame()

    try:
        return pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def numeric_column(
    df: pd.DataFrame,
    possible_columns: list[str],
    default: float = 0.0,
) -> pd.Series:
    for col in possible_columns:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(default)

    return pd.Series([default] * len(df), index=df.index)


def load_cash() -> float:
    cash_df = load_csv_if_exists(CASH_FILE)

    if cash_df.empty or "cash" not in cash_df.columns:
        return INITIAL_CASH

    cash = pd.to_numeric(cash_df["cash"], errors="coerce").dropna()

    if cash.empty:
        return INITIAL_CASH

    return float(cash.iloc[0])


def calculate_open_current_value(open_positions: pd.DataFrame) -> float:
    if open_positions.empty:
        return 0.0

    if "current_value_usd" in open_positions.columns:
        return float(
            pd.to_numeric(
                open_positions["current_value_usd"],
                errors="coerce",
            ).fillna(0.0).sum()
        )

    capital_at_risk = numeric_column(open_positions, ["capital_at_risk"])
    unrealized_pnl = numeric_column(
        open_positions,
        ["unrealized_pnl_usd", "unrealized_pnl"],
    )

    return float((capital_at_risk + unrealized_pnl).sum())


def calculate_unrealized_pnl(open_positions: pd.DataFrame) -> float:
    if open_positions.empty:
        return 0.0

    return float(
        numeric_column(
            open_positions,
            ["unrealized_pnl_usd", "unrealized_pnl"],
        ).sum()
    )


def calculate_realized_pnl(trade_history: pd.DataFrame) -> float:
    if trade_history.empty:
        return 0.0

    data = trade_history.copy()

    if "status" in data.columns:
        data = data[data["status"].astype(str).str.lower() == "closed"].copy()

    if data.empty:
        return 0.0

    return float(numeric_column(data, ["pnl_usd", "pnl"]).sum())


def calculate_expected_cash_from_history(trade_history: pd.DataFrame) -> float:
    """
    Reconstruct expected cash from trade history.

    Assumption:
    - open rows reduce cash by capital_at_risk
    - closed rows increase cash by exit_value_usd
    """

    if trade_history.empty:
        return INITIAL_CASH

    data = trade_history.copy()

    if "status" not in data.columns:
        return INITIAL_CASH

    data["status_normalized"] = data["status"].astype(str).str.lower().str.strip()

    open_rows = data[data["status_normalized"] == "open"].copy()
    closed_rows = data[data["status_normalized"] == "closed"].copy()

    total_entry_cash_used = 0.0
    total_exit_cash_returned = 0.0

    if not open_rows.empty:
        total_entry_cash_used = float(
            numeric_column(open_rows, ["capital_at_risk", "entry_value_usd"]).sum()
        )

    if not closed_rows.empty:
        if "exit_value_usd" in closed_rows.columns:
            total_exit_cash_returned = float(
                numeric_column(closed_rows, ["exit_value_usd"]).sum()
            )
        else:
            exit_price = numeric_column(closed_rows, ["exit_price_usd"])
            quantity = numeric_column(closed_rows, ["quantity"])
            total_exit_cash_returned = float((exit_price * quantity).sum())

    expected_cash = INITIAL_CASH - total_entry_cash_used + total_exit_cash_returned

    return float(expected_cash)


def generate_reconciliation_report() -> pd.DataFrame:
    trade_history = load_csv_if_exists(TRADE_HISTORY_FILE)
    open_positions = load_csv_if_exists(OPEN_POSITIONS_FILE)
    actual_cash = load_cash()

    expected_cash = calculate_expected_cash_from_history(trade_history)
    cash_difference = actual_cash - expected_cash

    open_current_value = calculate_open_current_value(open_positions)
    unrealized_pnl = calculate_unrealized_pnl(open_positions)
    realized_pnl = calculate_realized_pnl(trade_history)

    equity_estimate = actual_cash + open_current_value
    total_pnl_estimate = equity_estimate - INITIAL_CASH

    reconciliation_ok = abs(cash_difference) <= TOLERANCE

    summary = {
        "initial_cash": INITIAL_CASH,
        "actual_cash": actual_cash,
        "expected_cash_from_history": expected_cash,
        "cash_difference": cash_difference,
        "open_positions": int(len(open_positions)),
        "open_current_value": open_current_value,
        "unrealized_pnl": unrealized_pnl,
        "realized_pnl": realized_pnl,
        "equity_estimate": equity_estimate,
        "total_pnl_estimate": total_pnl_estimate,
        "reconciliation_ok": reconciliation_ok,
    }

    df = pd.DataFrame([summary])

    Path(RECONCILIATION_FILE).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RECONCILIATION_FILE, index=False)

    print("========== PAPER ACCOUNT RECONCILIATION ==========")
    print(f"Initial cash:             ${INITIAL_CASH:,.2f}")
    print(f"Actual cash:              ${actual_cash:,.2f}")
    print(f"Expected cash:            ${expected_cash:,.2f}")
    print(f"Cash difference:          ${cash_difference:,.2f}")
    print("--------------------------------------------------")
    print(f"Open positions:           {len(open_positions)}")
    print(f"Open current value:       ${open_current_value:,.2f}")
    print(f"Unrealized PnL:           ${unrealized_pnl:,.2f}")
    print(f"Realized PnL:             ${realized_pnl:,.2f}")
    print("--------------------------------------------------")
    print(f"Equity estimate:          ${equity_estimate:,.2f}")
    print(f"Total PnL estimate:       ${total_pnl_estimate:,.2f}")
    print(f"Reconciliation OK:        {reconciliation_ok}")
    print(f"Saved to:                 {RECONCILIATION_FILE}")
    print("==================================================")

    if not reconciliation_ok:
        print("WARNING: Paper account cash does not match trade history.")

    return df


if __name__ == "__main__":
    generate_reconciliation_report()