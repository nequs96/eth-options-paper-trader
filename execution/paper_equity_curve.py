"""
execution/paper_equity_curve.py

Appends one paper equity snapshot per scheduler cycle.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from execution.paper_trader import PaperTraderConfig
from execution.paper_account_reconciliation import load_csv_if_exists, load_cash, calculate_open_current_value, calculate_unrealized_pnl

EQUITY_CURVE_FILE = "outputs/paper_equity_curve.csv"


def generate_equity_curve(
    config: PaperTraderConfig | None = None,
    output_file: str = EQUITY_CURVE_FILE,
) -> pd.DataFrame:
    if config is None:
        config = PaperTraderConfig()

    open_positions = load_csv_if_exists(config.positions_file)
    cash = load_cash(config)
    open_value = calculate_open_current_value(open_positions)
    unrealized_pnl = calculate_unrealized_pnl(open_positions)
    equity = cash + open_value

    row = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp.utcnow().isoformat(),
                "cash": float(cash),
                "open_current_value": float(open_value),
                "unrealized_pnl": float(unrealized_pnl),
                "equity": float(equity),
                "open_positions": int(len(open_positions)) if not open_positions.empty else 0,
            }
        ]
    )

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    old = load_csv_if_exists(output_file)
    if old.empty:
        result = row
    else:
        result = pd.concat([old, row], ignore_index=True)

    result.to_csv(output_file, index=False)
    print("Equity curve updated.")
    return result


if __name__ == "__main__":
    generate_equity_curve()
