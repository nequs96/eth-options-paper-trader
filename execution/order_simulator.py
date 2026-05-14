from __future__ import annotations
from pathlib import Path
import pandas as pd
from models.execution_cost_model import add_execution_cost_estimates


def simulate_orders(candidates_file: str = 'outputs/live_backtest_candidates_surface_scored.csv', max_orders: int = 2):
    path = Path(candidates_file)
    candidates = pd.read_csv(path) if path.exists() and path.stat().st_size > 0 else pd.DataFrame()
    candidates = add_execution_cost_estimates(candidates)
    if candidates.empty:
        orders, fills = pd.DataFrame(), pd.DataFrame()
    else:
        if 'institutional_edge_score' in candidates.columns:
            candidates = candidates.sort_values('institutional_edge_score', ascending=False)
        chosen = candidates.head(max_orders).copy()
        now = pd.Timestamp.utcnow().isoformat()
        orders = chosen[[c for c in ['instrument_name','option_type','strike','days_to_expiry'] if c in chosen.columns]].copy()
        orders['order_id'] = [f'PAPER-{i + 1}' for i in range(len(orders))]
        orders['created_at'] = now
        orders['status'] = 'filled'
        orders['side'] = 'buy'
        fills = orders[['order_id','instrument_name','side']].copy()
        fills['filled_at'] = now
        fills['fill_price_usd'] = chosen.get('estimated_entry_price_usd', chosen.get('market_price_usd')).values
    Path('outputs').mkdir(exist_ok=True)
    orders.to_csv('outputs/paper_orders.csv', index=False)
    fills.to_csv('outputs/paper_fills.csv', index=False)
    print(f'Order simulation complete. orders={len(orders)} fills={len(fills)}')
    return orders, fills


if __name__ == '__main__':
    simulate_orders()
