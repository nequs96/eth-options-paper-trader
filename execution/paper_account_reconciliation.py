from __future__ import annotations
from pathlib import Path
import pandas as pd
from execution.paper_trader import PaperTraderConfig, load_csv_if_exists, load_cash, open_only, numeric_series


def closed_only(data: pd.DataFrame) -> pd.DataFrame:
    return data if data.empty or 'status' not in data.columns else data[data['status'].astype(str).str.lower().eq('closed')]


def calculate_open_cost_basis(data: pd.DataFrame) -> float:
    return 0.0 if data.empty else float(numeric_series(open_only(data), 'capital_at_risk', 0.0).sum())


def calculate_open_current_value(data: pd.DataFrame) -> float:
    data = open_only(data)
    return 0.0 if data.empty else float(numeric_series(data, 'current_value_usd', 0.0).sum())


def calculate_unrealized_pnl(data: pd.DataFrame) -> float:
    data = open_only(data)
    return 0.0 if data.empty else float(numeric_series(data, 'unrealized_pnl_usd', 0.0).sum())


def calculate_realized_pnl(history: pd.DataFrame) -> float:
    return 0.0 if history.empty else float(numeric_series(closed_only(history), 'pnl_usd', 0.0).sum())


def generate_reconciliation_report(config: PaperTraderConfig | None = None, output_file: str = 'outputs/paper_account_reconciliation.csv') -> pd.DataFrame:
    config = config or PaperTraderConfig()
    positions = open_only(load_csv_if_exists(config.positions_file))
    cash = load_cash(config)
    value = calculate_open_current_value(positions)
    duplicates = int(positions['instrument_name'].duplicated().sum()) if not positions.empty and 'instrument_name' in positions.columns else 0
    ok = duplicates == 0 and cash >= -0.1
    report = pd.DataFrame([{
        'timestamp': pd.Timestamp.utcnow().isoformat(),
        'status': 'ok' if ok else 'hard_fail',
        'reconciliation_ok': ok,
        'actual_cash': cash,
        'equity_estimate': cash + value,
        'open_positions': len(positions),
        'duplicate_open_instruments': duplicates,
    }])
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_file, index=False)
    print(f"Reconciliation: {report.iloc[-1]['status']}  equity={cash + value:,.2f}  open={len(positions)}")
    return report
