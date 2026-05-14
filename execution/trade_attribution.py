from __future__ import annotations
from pathlib import Path
import pandas as pd


def run_trade_attribution(history_file: str = 'outputs/paper_trade_history.csv') -> pd.DataFrame:
    path = Path(history_file)
    history = pd.read_csv(path) if path.exists() and path.stat().st_size > 0 else pd.DataFrame()
    output = history.copy() if not history.empty else pd.DataFrame()
    if not output.empty:
        output['delta_component_estimate'] = 0.0
        output['vega_component_estimate'] = 0.0
        output['theta_component_estimate'] = 0.0
        output['residual_component_estimate'] = pd.to_numeric(output.get('pnl_usd', 0.0), errors='coerce').fillna(0.0)
    Path('outputs').mkdir(exist_ok=True)
    output.to_csv('outputs/trade_attribution.csv', index=False)
    print(f'Trade attribution complete. rows={len(output)}')
    return output


if __name__ == '__main__':
    run_trade_attribution()
