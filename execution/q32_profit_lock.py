from __future__ import annotations
import pandas as pd
from execution.q32_config import Q32ProfitRobustConfig
from execution.q32_io import load_csv, atomic_csv, append_event, num


def build_profit_lock_actions(cfg: Q32ProfitRobustConfig | None = None) -> pd.DataFrame:
    cfg = cfg or Q32ProfitRobustConfig(); pos = load_csv('outputs/paper_open_positions.csv')
    if pos.empty:
        out = pd.DataFrame(); atomic_csv('outputs/q32_profit_lock_actions.csv', out); return out
    open_pos = pos[pos['status'].astype(str).str.lower().eq('open')].copy() if 'status' in pos.columns else pos.copy()
    if open_pos.empty:
        out = pd.DataFrame(); atomic_csv('outputs/q32_profit_lock_actions.csv', out); return out
    pnl, high, theta = num(open_pos,'unrealized_pnl_pct',0), num(open_pos,'highest_profit_pct',0), num(open_pos,'theta',0)
    actions=[]
    for idx,r in open_pos.iterrows():
        p,h,th = float(pnl.loc[idx]), float(high.loc[idx]), float(theta.loc[idx]); giveback=h-p
        action='HOLD'; reason=''
        if p >= cfg.profit_take_full_pct: action, reason = 'CLOSE_FULL_PROFIT_TARGET', f'pnl_pct_ge_{cfg.profit_take_full_pct}'
        elif p >= cfg.profit_take_partial_pct and giveback >= cfg.profit_giveback_max_pct: action, reason = 'CLOSE_FULL_GIVEBACK_PROTECTION', 'large_profit_giveback'
        elif p >= cfg.profit_lock_trigger_pct and th < 0: action, reason = 'LOCK_PROFIT_REVIEW_CLOSE', 'profitable_negative_theta_position'
        if action != 'HOLD':
            row=r.to_dict(); row.update({'q32_action':action,'q32_reason':reason,'q32_unrealized_pnl_pct':p,'q32_highest_profit_pct':h,'q32_giveback_pct':giveback}); actions.append(row)
    out=pd.DataFrame(actions); atomic_csv('outputs/q32_profit_lock_actions.csv', out); append_event('outputs/q32_profit_lock_log.csv', {'timestamp':pd.Timestamp.utcnow().isoformat(),'actions':len(out)}); print(f'Q32 profit lock actions={len(out)}'); return out


def execute_profit_lock_paper(auto_execute: bool = False, cfg: Q32ProfitRobustConfig | None = None) -> pd.DataFrame:
    actions=build_profit_lock_actions(cfg)
    if actions.empty or not auto_execute: return actions
    pos,hist=load_csv('outputs/paper_open_positions.csv'),load_csv('outputs/paper_trade_history.csv'); close_names=set(actions['instrument_name'].astype(str)) if 'instrument_name' in actions.columns else set()
    keep=[]; closed=[]; now=pd.Timestamp.utcnow().isoformat()
    for _,r in pos.iterrows():
        if str(r.get('instrument_name')) in close_names and str(r.get('status','open')).lower()=='open':
            row=r.to_dict(); exit_price=float(row.get('bid_price_usd') or row.get('current_price_usd') or row.get('market_price_usd') or row.get('entry_price_usd') or 0); qty=float(row.get('quantity') or 0); entry=float(row.get('entry_price_usd') or exit_price or 0)
            row.update({'status':'closed','closed_at':now,'close_reason':'q32_profit_lock','exit_price_usd':exit_price,'exit_value_usd':qty*exit_price,'proceeds_usd':qty*exit_price,'pnl_usd':qty*(exit_price-entry),'pnl_pct':((exit_price-entry)/entry) if entry else 0}); closed.append(row)
        else: keep.append(r.to_dict())
    atomic_csv('outputs/paper_open_positions.csv', pd.DataFrame(keep)); new_hist=pd.concat([hist,pd.DataFrame(closed)], ignore_index=True) if not hist.empty else pd.DataFrame(closed); atomic_csv('outputs/paper_trade_history.csv', new_hist); atomic_csv('outputs/q32_profit_lock_closed.csv', pd.DataFrame(closed)); print(f'Q32 profit lock executed closes={len(closed)}'); return pd.DataFrame(closed)

if __name__ == '__main__': build_profit_lock_actions()
