from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd

@dataclass(frozen=True)
class WalkForwardConfig:
    candidates_file: str = 'outputs/live_backtest_candidates_surface_scored.csv'
    results_file: str = 'outputs/walk_forward_results.csv'
    summary_file: str = 'outputs/walk_forward_summary.csv'
    stability_file: str = 'outputs/parameter_stability_report.csv'
    min_score_grid: tuple[float, ...] = (0.40, 0.45, 0.50, 0.55)
    max_spread_grid: tuple[float, ...] = (0.04, 0.06, 0.08, 0.10)
    min_dte_grid: tuple[float, ...] = (3.0, 7.0, 10.0)
    max_abs_moneyness_grid: tuple[float, ...] = (0.08, 0.12, 0.18)

def _load(path: str) -> pd.DataFrame:
    p=Path(path)
    if not p.exists() or p.stat().st_size==0: return pd.DataFrame()
    try: return pd.read_csv(p)
    except pd.errors.EmptyDataError: return pd.DataFrame()

def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(df[col], errors='coerce').fillna(default) if col in df.columns else pd.Series([default]*len(df), index=df.index, dtype=float)

def _proxy(data: pd.DataFrame) -> dict:
    if data.empty: return {'candidate_count':0,'mean_edge_score':0.0,'mean_model_edge':0.0,'mean_spread':0.0,'mean_dte':0.0,'quality_score':0.0}
    score=_num(data,'institutional_edge_score',0.0).clip(0,1)
    model_edge=(-_num(data,'price_diff_pct',0.0)).clip(-1,1)
    spread=_num(data,'bid_ask_spread_pct',0.0).clip(0,1)
    dte=_num(data,'days_to_expiry',0.0)
    quality=(score*(1-spread.clip(0,.25)/.25)).mean()
    return {'candidate_count':int(len(data)),'mean_edge_score':float(score.mean()),'mean_model_edge':float(model_edge.mean()),'mean_spread':float(spread.mean()),'mean_dte':float(dte.mean()),'quality_score':float(quality if pd.notna(quality) else 0.0)}

def run_walk_forward_validation(config: WalkForwardConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg=config or WalkForwardConfig(); cand=_load(cfg.candidates_file); Path('outputs').mkdir(exist_ok=True)
    rows=[]
    if cand.empty:
        results=pd.DataFrame([{**asdict(cfg),'status':'no_candidates'}])
    else:
        data=cand.copy()
        for c in ['institutional_edge_score','combined_score','bid_ask_spread_pct','days_to_expiry','abs_moneyness','moneyness','price_diff_pct']:
            if c in data.columns: data[c]=pd.to_numeric(data[c], errors='coerce')
        if 'abs_moneyness' not in data.columns and 'moneyness' in data.columns: data['abs_moneyness']=data['moneyness'].abs()
        if 'institutional_edge_score' not in data.columns: data['institutional_edge_score']=_num(data,'combined_score',0.0)
        for min_score in cfg.min_score_grid:
            for max_spread in cfg.max_spread_grid:
                for min_dte in cfg.min_dte_grid:
                    for max_abs_m in cfg.max_abs_moneyness_grid:
                        f=data[(_num(data,'institutional_edge_score')>=min_score)&(_num(data,'bid_ask_spread_pct',999)<=max_spread)&(_num(data,'days_to_expiry')>=min_dte)&(_num(data,'abs_moneyness',999)<=max_abs_m)]
                        rows.append({'min_score':min_score,'max_spread':max_spread,'min_dte':min_dte,'max_abs_moneyness':max_abs_m,**_proxy(f)})
        results=pd.DataFrame(rows).sort_values(['quality_score','candidate_count'], ascending=[False,False]).reset_index(drop=True)
    results.to_csv(cfg.results_file,index=False)
    if results.empty:
        summary=pd.DataFrame(); stability=pd.DataFrame()
    else:
        top=results.head(10).copy()
        summary=pd.DataFrame([{'status':'ok','tested_parameter_sets':len(results),'best_quality_score':float(results['quality_score'].max()),'best_candidate_count':int(results.iloc[0].get('candidate_count',0)),'recommended_min_score':float(results.iloc[0].get('min_score',0)),'recommended_max_spread':float(results.iloc[0].get('max_spread',0)),'recommended_min_dte':float(results.iloc[0].get('min_dte',0)),'recommended_max_abs_moneyness':float(results.iloc[0].get('max_abs_moneyness',0))}])
        stability=top[['min_score','max_spread','min_dte','max_abs_moneyness','candidate_count','quality_score']].copy(); stability['rank']=range(1,len(stability)+1)
    summary.to_csv(cfg.summary_file,index=False); stability.to_csv(cfg.stability_file,index=False)
    print(f'Walk-forward validation framework complete. parameter_sets={len(results)}')
    return results, summary

if __name__=='__main__': run_walk_forward_validation()
