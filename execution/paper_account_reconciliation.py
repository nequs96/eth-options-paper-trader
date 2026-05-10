"""Paper account reconciliation with hard/soft failure separation."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from execution.paper_trader import PaperTraderConfig

RECONCILIATION_FILE = "outputs/paper_account_reconciliation.csv"
TOLERANCE = 0.10


def load_csv_if_exists(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def numeric_column(df: pd.DataFrame, possible_columns: list[str], default: float = 0.0) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    for column in possible_columns:
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce").fillna(default)
    return pd.Series([default] * len(df), index=df.index, dtype=float)


def open_only(positions: pd.DataFrame) -> pd.DataFrame:
    if positions.empty:
        return positions.copy()
    data = positions.copy()
    if "status" in data.columns:
        data = data[data["status"].astype(str).str.lower().eq("open")]
    return data.reset_index(drop=True)


def closed_only(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return history.copy()
    data = history.copy()
    if "status" in data.columns:
        data = data[data["status"].astype(str).str.lower().eq("closed")]
    return data.reset_index(drop=True)


def load_cash(config: PaperTraderConfig) -> float:
    data = load_csv_if_exists(config.initial_cash_file)
    if data.empty:
        return float(config.initial_cash)
    for column in ["cash", "paper_cash", "current_cash"]:
        if column in data.columns:
            values = pd.to_numeric(data[column], errors="coerce").dropna()
            if not values.empty:
                return float(values.iloc[-1])
    return float(config.initial_cash)


def calculate_open_cost_basis(open_positions: pd.DataFrame) -> float:
    data = open_only(open_positions)
    if data.empty:
        return 0.0
    cost = numeric_column(data, ["capital_at_risk", "entry_cost_usd", "cost_usd"], 0.0)
    if float(cost.sum()) > 0:
        return float(cost.sum())
    return float((numeric_column(data, ["entry_price_usd"], 0.0) * numeric_column(data, ["quantity"], 0.0)).sum())


def calculate_open_current_value(open_positions: pd.DataFrame) -> float:
    data = open_only(open_positions)
    if data.empty:
        return 0.0
    value = numeric_column(data, ["current_value_usd", "market_value_usd"], 0.0)
    if float(value.sum()) > 0:
        return float(value.sum())
    return float((numeric_column(data, ["current_price_usd", "entry_price_usd"], 0.0) * numeric_column(data, ["quantity"], 0.0)).sum())


def calculate_unrealized_pnl(open_positions: pd.DataFrame) -> float:
    data = open_only(open_positions)
    if data.empty:
        return 0.0
    pnl = numeric_column(data, ["unrealized_pnl_usd", "unrealized_pnl"], 0.0)
    if abs(float(pnl.sum())) > 0:
        return float(pnl.sum())
    return float(((numeric_column(data, ["current_price_usd"], 0.0) - numeric_column(data, ["entry_price_usd"], 0.0)) * numeric_column(data, ["quantity"], 0.0)).sum())


def calculate_realized_pnl(history: pd.DataFrame) -> float:
    data = closed_only(history)
    if data.empty:
        return 0.0
    pnl = numeric_column(data, ["pnl_usd", "pnl"], 0.0)
    if abs(float(pnl.sum())) > 0:
        return float(pnl.sum())
    proceeds = numeric_column(data, ["exit_value_usd", "proceeds_usd"], 0.0)
    if float(proceeds.sum()) == 0.0:
        proceeds = numeric_column(data, ["exit_price_usd", "current_price_usd"], 0.0) * numeric_column(data, ["quantity"], 0.0)
    cost = numeric_column(data, ["capital_at_risk", "entry_cost_usd", "cost_usd"], 0.0)
    if float(cost.sum()) == 0.0:
        cost = numeric_column(data, ["entry_price_usd"], 0.0) * numeric_column(data, ["quantity"], 0.0)
    return float(proceeds.sum() - cost.sum())


def calculate_expected_cash_from_history(config: PaperTraderConfig, history: pd.DataFrame, open_positions: pd.DataFrame) -> float:
    expected = float(config.initial_cash) - calculate_open_cost_basis(open_positions)
    closed = closed_only(history)
    if not closed.empty:
        cost = numeric_column(closed, ["capital_at_risk", "entry_cost_usd", "cost_usd"], 0.0)
        if float(cost.sum()) == 0.0:
            cost = numeric_column(closed, ["entry_price_usd"], 0.0) * numeric_column(closed, ["quantity"], 0.0)
        proceeds = numeric_column(closed, ["exit_value_usd", "proceeds_usd"], 0.0)
        if float(proceeds.sum()) == 0.0:
            proceeds = numeric_column(closed, ["exit_price_usd", "current_price_usd"], 0.0) * numeric_column(closed, ["quantity"], 0.0)
        expected -= float(cost.sum())
        expected += float(proceeds.sum())
    return float(expected)


def generate_reconciliation_report(config: PaperTraderConfig | None = None, output_file: str = RECONCILIATION_FILE) -> pd.DataFrame:
    if config is None:
        config = PaperTraderConfig()
    history = load_csv_if_exists(config.trade_history_file)
    positions_all = load_csv_if_exists(config.positions_file)
    positions = open_only(positions_all)
    cash = load_cash(config)
    expected_cash = calculate_expected_cash_from_history(config, history, positions)
    cash_difference = cash - expected_cash
    open_value = calculate_open_current_value(positions)
    open_cost = calculate_open_cost_basis(positions)
    realized = calculate_realized_pnl(history)
    unrealized = calculate_unrealized_pnl(positions)
    duplicates = 0
    if not positions.empty and "instrument_name" in positions.columns:
        names = positions["instrument_name"].astype(str).str.strip()
        duplicates = int(names[names.ne("")].duplicated().sum())
    hard_errors = bool(cash < -TOLERANCE or open_value < -TOLERANCE or open_cost < -TOLERANCE or duplicates > 0)
    ledger_cash_match = abs(cash_difference) <= TOLERANCE
    status = "hard_fail" if hard_errors else "ok" if ledger_cash_match else "soft_ledger_mismatch"
    report = pd.DataFrame([{
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "status": status,
        "reconciliation_ok": not hard_errors,
        "ledger_cash_match": ledger_cash_match,
        "actual_cash": float(cash),
        "expected_cash_from_history": float(expected_cash),
        "cash_difference": float(cash_difference),
        "open_cost_basis": float(open_cost),
        "open_current_value": float(open_value),
        "realized_pnl": float(realized),
        "unrealized_pnl": float(unrealized),
        "actual_total_pnl": float(realized + unrealized),
        "equity_estimate": float(cash + open_value),
        "open_positions": int(len(positions)),
        "closed_trades": int(len(closed_only(history))),
        "duplicate_open_instruments": duplicates,
    }])
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_file, index=False)
    row = report.iloc[-1]
    print(f"Reconciliation: {row['status']} | cash=${row['actual_cash']:,.2f} | equity=${row['equity_estimate']:,.2f} | open={int(row['open_positions'])}")
    return report


if __name__ == "__main__":
    generate_reconciliation_report()
