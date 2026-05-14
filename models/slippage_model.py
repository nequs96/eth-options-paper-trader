from __future__ import annotations
import pandas as pd
from execution.common_io import num

def add_slippage_estimates(df: pd.DataFrame, side: str='buy') -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    out=df.copy(); mark=num(out,'market_price_usd'); bid=num(out,'bid_price_usd'); ask=num(out,'ask_price_usd'); spread=num(out,'bid_ask_spread_pct')
    out['realistic_entry_price_usd']=ask.where(ask>0, mark*(1+spread/2))
    out['realistic_exit_price_usd']=bid.where(bid>0, mark*(1-spread/2))
    out['estimated_round_trip_cost_usd']=(out['realistic_entry_price_usd']-out['realistic_exit_price_usd']).clip(lower=0)
    out['execution_reject_reason']=''
    out.loc[spread>0.10,'execution_reject_reason']='spread_too_wide_for_realistic_execution'
    out.loc[(out['realistic_entry_price_usd']<=0)|(out['realistic_exit_price_usd']<=0),'execution_reject_reason']='missing_bid_ask'
    return out
