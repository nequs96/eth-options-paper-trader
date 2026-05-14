from __future__ import annotations
from pathlib import Path
import pandas as pd
from execution.common_io import load_csv, num

def generate_risk_budget_sizing(candidates_file='outputs/optimized_trade_list.csv', greeks_file='outputs/portfolio_greeks.csv', output_file='outputs/risk_budget_sizing_report.csv'):
    c=load_csv(candidates_file); g=load_csv(greeks_file); Path('outputs').mkdir(exist_ok=True)
    if c.empty:
        out=pd.DataFrame(); out.to_csv(output_file,index=False); return out
    theta_now=abs(float(g.iloc[-1].get('net_theta',0))) if not g.empty else 0.0
    theta_remaining=max(50-theta_now,0)
    out=c.copy(); theta_abs=abs(num(out,'theta',1.0)).replace(0,1.0); price=num(out,'market_price_usd',1.0).replace(0,1.0)
    out['theta_budget_quantity_hint']=(theta_remaining/theta_abs).clip(lower=0)
    out['cash_budget_quantity_hint']=(100.0/price).clip(lower=0)
    out['risk_budget_quantity_hint']=out[['theta_budget_quantity_hint','cash_budget_quantity_hint']].min(axis=1)
    out.to_csv(output_file,index=False); print(f'Risk-budget sizing complete. rows={len(out)} theta_remaining={theta_remaining:.2f}'); return out
if __name__=='__main__': generate_risk_budget_sizing()
