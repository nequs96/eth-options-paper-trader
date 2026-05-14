from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd

@dataclass(frozen=True)
class SmartExitConfig:
    positions_file: str = 'outputs/paper_open_positions.csv'
    candidates_file: str = 'outputs/live_backtest_candidates_surface_scored.csv'
    surface_file: str = 'outputs/iv_surface_diagnostics.csv'
    risk_breaches_file: str = 'outputs/risk_limit_breaches.csv'
    output_file: str = 'outputs/smart_exit_recommendations.csv'
    min_days_to_expiry_exit: float = 2.0
    hard_stop_loss_pct: float = -0.30
    hard_take_profit_pct: float = 0.80
    profit_giveback_activation_pct: float = 0.35
    profit_giveback_keep_pct: float = 0.15
    min_surface_score_hold: float = 0.35
    min_edge_score_hold: float = 0.35
    force_review_on_risk_breach: bool = True

def _load(path):
    p=Path(path)
    if not p.exists() or p.stat().st_size==0: return pd.DataFrame()
    try: return pd.read_csv(p)
    except pd.errors.EmptyDataError: return pd.DataFrame()

def _sf(v,d=0.0):
    try: x=float(v)
    except Exception: return d
    return x if pd.notna(x) else d

def generate_smart_exit_recommendations(config: SmartExitConfig | None = None) -> pd.DataFrame:
    cfg=config or SmartExitConfig(); pos=_load(cfg.positions_file); cand=_load(cfg.candidates_file); surf=_load(cfg.surface_file); breaches=_load(cfg.risk_breaches_file); Path('outputs').mkdir(exist_ok=True)
    if pos.empty:
        out=pd.DataFrame(columns=['instrument_name','exit_action','exit_reason','priority']); out.to_csv(cfg.output_file,index=False); print('Smart exit recommendations complete. rows=0'); return out
    if 'status' in pos.columns: pos=pos[pos['status'].astype(str).str.lower().eq('open')].copy()
    cmap=cand.set_index('instrument_name').to_dict('index') if not cand.empty and 'instrument_name' in cand.columns else {}
    smap=surf.set_index('instrument_name').to_dict('index') if not surf.empty and 'instrument_name' in surf.columns else {}
    rows=[]; has_breach=not breaches.empty
    for _,r in pos.iterrows():
        inst=str(r.get('instrument_name','')); c=cmap.get(inst,{}); sv=smap.get(inst,{})
        dte=_sf(r.get('days_to_expiry'),999); pnl=_sf(r.get('unrealized_pnl_pct'),0); high=_sf(r.get('highest_profit_pct'),pnl)
        edge=_sf(c.get('institutional_edge_score',r.get('institutional_edge_score',.5)),.5); surface=_sf(sv.get('surface_relative_value_score',r.get('surface_relative_value_score',.5)),.5)
        reasons=[]; action='HOLD'; priority='LOW'
        def mark(reason, act='REVIEW_EXIT', pr='MEDIUM'):
            nonlocal action, priority
            reasons.append(reason); action=act if action=='HOLD' else action; priority='HIGH' if priority=='HIGH' or pr=='HIGH' else pr
        if dte<=cfg.min_days_to_expiry_exit: mark('near_expiry','REVIEW_EXIT','HIGH')
        if pnl<=cfg.hard_stop_loss_pct: mark('hard_stop_loss','REVIEW_EXIT','HIGH')
        if pnl>=cfg.hard_take_profit_pct: mark('hard_take_profit','REVIEW_TAKE_PROFIT','HIGH')
        if high>=cfg.profit_giveback_activation_pct and pnl<=cfg.profit_giveback_keep_pct: mark('profit_giveback')
        if surface<cfg.min_surface_score_hold: mark('surface_edge_deteriorated')
        if edge<cfg.min_edge_score_hold: mark('optimizer_edge_deteriorated')
        if cfg.force_review_on_risk_breach and has_breach: mark('portfolio_risk_limit_breach','REVIEW_PORTFOLIO_RISK_REDUCTION','HIGH')
        rows.append({'instrument_name':inst,'option_type':r.get('option_type'),'strike':r.get('strike'),'days_to_expiry':dte,'unrealized_pnl_pct':pnl,'highest_profit_pct':high,'institutional_edge_score':edge,'surface_relative_value_score':surface,'exit_action':action,'exit_reason':';'.join(reasons) if reasons else 'none','priority':priority})
    out=pd.DataFrame(rows); order={'HIGH':0,'MEDIUM':1,'LOW':2}; out['_s']=out['priority'].map(order).fillna(9); out=out.sort_values(['_s','days_to_expiry']).drop(columns=['_s']).reset_index(drop=True)
    out.to_csv(cfg.output_file,index=False); print(f'Smart exit recommendations complete. rows={len(out)} review={int((out["exit_action"]!="HOLD").sum())}')
    return out

if __name__=='__main__': generate_smart_exit_recommendations()
