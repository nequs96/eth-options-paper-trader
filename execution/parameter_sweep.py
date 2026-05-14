from __future__ import annotations
from itertools import product
from pathlib import Path
import pandas as pd

GRID = {'min_mci_to_accept': [0.30, 0.35, 0.40], 'max_bid_ask_spread_pct': [0.20, 0.35, 0.50], 'min_market_price_usd': [1, 5, 10]}


def run_parameter_sweep(candidates_file: str = 'outputs/live_backtest_candidates.csv') -> pd.DataFrame:
    path = Path(candidates_file)
    candidates = pd.read_csv(path) if path.exists() and path.stat().st_size > 0 else pd.DataFrame()
    rows = []
    keys = list(GRID)
    for values in product(*[GRID[key] for key in keys]):
        params = dict(zip(keys, values))
        data = candidates.copy()
        if not data.empty:
            if 'mci' in data.columns:
                data = data[pd.to_numeric(data['mci'], errors='coerce').fillna(0) >= params['min_mci_to_accept']]
            if 'bid_ask_spread_pct' in data.columns:
                data = data[pd.to_numeric(data['bid_ask_spread_pct'], errors='coerce').fillna(999) <= params['max_bid_ask_spread_pct']]
            if 'market_price_usd' in data.columns:
                data = data[pd.to_numeric(data['market_price_usd'], errors='coerce').fillna(0) >= params['min_market_price_usd']]
        rows.append({**params, 'surviving_candidates': len(data)})
    output = pd.DataFrame(rows)
    Path('outputs').mkdir(exist_ok=True)
    output.to_csv('outputs/parameter_sweep_results.csv', index=False)
    output.sort_values('surviving_candidates', ascending=False).head(10).to_csv('outputs/parameter_sweep_summary.csv', index=False)
    print('Parameter sweep complete.')
    return output


if __name__ == '__main__':
    run_parameter_sweep()
