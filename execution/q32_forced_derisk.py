from __future__ import annotations
import pandas as pd
from execution.q32_config import Q32ProfitRobustConfig
from execution.q32_io import load_csv, atomic_csv, append_event, num


def build_forced_derisk_actions(cfg: Q32ProfitRobustConfig | None = None) -> pd.DataFrame:
    cfg = cfg or Q32ProfitRobustConfig(); risk=load_csv('outputs/portfolio_limit_report.csv'); pos=load_csv('outputs/paper_open_positions.csv')
    if risk.empty or pos.empty:
        out=pd.DataFrame(); atomic_csv('outputs/q32_forced_derisk_actions.csv', out); return out
    latest=risk.iloc[-1]
    if str(latest.get('risk_status','')).upper()!='BREACH':
        out=pd.DataFrame(); atomic_csv('outputs/q32_forced_derisk_actions.csv', out); return out
    open_pos=pos[pos['status'].astype(str).str.lower().eq('open')].copy() if 'status' in pos.columns else pos.copy()
    if open_pos.empty:
        out=pd.DataFrame(); atomic_csv('outputs/q32_forced_derisk_actions.csv', out); return out
    pnl,theta,dte,spread=num(open_pos,'unrealized_pnl_pct',0),num(open_pos,'theta',0),num(open_pos,'days_to_expiry',999).clip(lower=.1),num(open_pos,'bid_ask_spread_pct',.1).clip(0,1)
    open_pos['q32_derisk_score']=pnl.clip(-.25,1.5)*.45+(-theta).clip(lower=0)*.08+(1/dte)*.25+spread*.20
    open_pos['q32_derisk_reason']='portfolio_breach_profit_or_theta_reduction'; open_pos['q32_action']='CLOSE_TO_REDUCE_RISK'
    out=open_pos.sort_values('q32_derisk_score', ascending=False).head(2).copy(); atomic_csv('outputs/q32_forced_derisk_actions.csv', out); append_event('outputs/q32_forced_derisk_log.csv', {'timestamp':pd.Timestamp.utcnow().isoformat(),'actions':len(out),'risk_status':latest.get('risk_status'),'flat_7d_loss':latest.get('flat_7d_loss'),'net_theta':latest.get('net_theta')}); print(f'Q32 forced derisk actions={len(out)}'); return out


def execute_forced_derisk_paper(auto_execute: bool = False, cfg: Q32ProfitRobustConfig | None = None) -> pd.DataFrame:
    actions=build_forced_derisk_actions(cfg)
    if actions.empty or not auto_execute: return actions
    pos,hist=load_csv('outputs/paper_open_positions.csv'),load_csv('outputs/paper_trade_history.csv'); close_names=set(actions['instrument_name'].astype(str)) if 'instrument_name' in actions.columns else set(); keep=[]; closed=[]; now=pd.Timestamp.utcnow().isoformat()
    for _,r in pos.iterrows():
        if str(r.get('instrument_name')) in close_names and str(r.get('status','open')).lower()=='open':
            row=r.to_dict(); exit_price=float(row.get('bid_price_usd') or row.get('current_price_usd') or row.get('market_price_usd') or row.get('entry_price_usd') or 0); qty=float(row.get('quantity') or 0); entry=float(row.get('entry_price_usd') or exit_price or 0); row.update({'status':'closed','closed_at':now,'close_reason':'q32_forced_derisk','exit_price_usd':exit_price,'exit_value_usd':qty*exit_price,'proceeds_usd':qty*exit_price,'pnl_usd':qty*(exit_price-entry),'pnl_pct':((exit_price-entry)/entry) if entry else 0}); closed.append(row)
        else: keep.append(r.to_dict())
    atomic_csv('outputs/paper_open_positions.csv', pd.DataFrame(keep)); new_hist=pd.concat([hist,pd.DataFrame(closed)], ignore_index=True) if not hist.empty else pd.DataFrame(closed); atomic_csv('outputs/paper_trade_history.csv', new_hist); atomic_csv('outputs/q32_forced_derisk_closed.csv', pd.DataFrame(closed)); print(f'Q32 forced derisk executed closes={len(closed)}'); return pd.DataFrame(closed)

if __name__ == '__main__': build_forced_derisk_actions()
