from __future__ import annotations
from pathlib import Path
import pandas as pd
from execution.common_io import load_csv
from models.slippage_model import add_slippage_estimates

def simulate_realistic_orders(candidates_file='outputs/optimized_trade_list.csv', orders_file='outputs/realistic_paper_orders.csv', quality_file='outputs/execution_quality_report.csv'):
    df=load_csv(candidates_file); Path('outputs').mkdir(exist_ok=True)
    ex=add_slippage_estimates(df)
    if ex.empty:
        ex.to_csv(orders_file,index=False); ex.to_csv(quality_file,index=False); print('Realistic order simulator: no candidates.'); return ex
    orders=ex[ex['execution_reject_reason'].astype(str).eq('')].copy(); orders['order_status']='simulated_fill'; orders['fill_price_usd']=orders['realistic_entry_price_usd']
    orders.to_csv(orders_file,index=False); ex.to_csv(quality_file,index=False)
    print(f'Realistic order simulation complete. fills={len(orders)} rejected={len(ex)-len(orders)}'); return orders
if __name__=='__main__': simulate_realistic_orders()
