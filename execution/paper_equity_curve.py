from __future__ import annotations
from pathlib import Path
import pandas as pd
from execution.paper_trader import PaperTraderConfig, load_csv_if_exists, load_cash
from execution.paper_account_reconciliation import calculate_open_current_value, calculate_unrealized_pnl


def generate_equity_curve(config: PaperTraderConfig | None = None, output_file: str = 'outputs/paper_equity_curve.csv') -> pd.DataFrame:
    config = config or PaperTraderConfig()
    positions = load_csv_if_exists(config.positions_file)
    cash = load_cash(config)
    value = calculate_open_current_value(positions)
    row = pd.DataFrame([{'timestamp': pd.Timestamp.utcnow().isoformat(), 'cash': cash, 'open_current_value': value, 'unrealized_pnl': calculate_unrealized_pnl(positions), 'equity': cash + value, 'open_positions': len(positions)}])
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    old = load_csv_if_exists(output_file)
    out = row if old.empty else pd.concat([old, row], ignore_index=True)
    out.to_csv(output_file, index=False)
    return out
