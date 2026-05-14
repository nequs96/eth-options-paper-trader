from __future__ import annotations
from pathlib import Path
import pandas as pd


def _load(path):
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _num(df, col, default=0.0):
    return pd.to_numeric(df[col], errors='coerce').fillna(default) if col in df.columns else pd.Series([default] * len(df), index=df.index)


def explain(row):
    return f"{row.get('instrument_name','candidate')}: model_edge={float(row.get('price_diff_pct',0)):.2%}; vol_spread={float(row.get('volatility_spread',0)):.2%}; surface_score={float(row.get('surface_relative_value_score',0)):.2f}; spread={float(row.get('bid_ask_spread_pct',0)):.2%}"


def score_candidates_with_surface(candidates: pd.DataFrame, surface: pd.DataFrame):
    if candidates is None or candidates.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    df = candidates.copy()
    if surface is not None and not surface.empty and 'instrument_name' in surface.columns:
        keep = [c for c in ['instrument_name','same_expiry_iv_percentile','surface_iv_zscore','surface_relative_value_score','term_bucket'] if c in surface.columns]
        df = df.merge(surface[keep], on='instrument_name', how='left')
    model = (-_num(df, 'price_diff_pct')).clip(0, 1)
    surf = _num(df, 'surface_relative_value_score', 0.5).clip(0, 1)
    liq = (1 - _num(df, 'bid_ask_spread_pct') / 0.35).clip(0, 1)
    vol = (-_num(df, 'volatility_spread')).clip(0, 1)
    base = _num(df, 'combined_score').clip(0, 1)
    df['institutional_edge_score'] = (0.30 * model + 0.25 * surf + 0.15 * liq + 0.15 * vol + 0.15 * base).clip(0, 1)
    df['candidate_explanation'] = df.apply(explain, axis=1)
    accepted = df[df['institutional_edge_score'] >= 0.35].sort_values('institutional_edge_score', ascending=False).reset_index(drop=True)
    rejected = df[df['institutional_edge_score'] < 0.35].copy()
    if not rejected.empty:
        rejected['surface_reject_reason'] = 'institutional_edge_score_below_threshold'
    explanations = df[[c for c in ['instrument_name','institutional_edge_score','candidate_explanation'] if c in df.columns]]
    return accepted, rejected, explanations


def run_surface_candidate_scorer(candidates_file='outputs/live_backtest_candidates.csv', surface_file='outputs/iv_surface_diagnostics.csv'):
    accepted, rejected, explanations = score_candidates_with_surface(_load(candidates_file), _load(surface_file))
    Path('outputs').mkdir(exist_ok=True)
    accepted.to_csv('outputs/live_backtest_candidates_surface_scored.csv', index=False)
    rejected.to_csv('outputs/live_backtest_candidates_surface_rejected.csv', index=False)
    explanations.to_csv('outputs/candidate_explanations.csv', index=False)
    print(f'Surface candidate scoring complete. accepted={len(accepted)} rejected={len(rejected)}')
    return accepted


if __name__ == '__main__':
    run_surface_candidate_scorer()
