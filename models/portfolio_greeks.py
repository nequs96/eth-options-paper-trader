from __future__ import annotations
from pathlib import Path
import pandas as pd


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    if col not in df.columns:
        return pd.Series([0.0] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(float)


def run_portfolio_greeks(positions_file: str = 'outputs/paper_open_positions.csv') -> pd.DataFrame:
    path = Path(positions_file)
    df = pd.read_csv(path) if path.exists() and path.stat().st_size > 0 else pd.DataFrame()
    if not df.empty and 'status' in df.columns:
        df = df[df['status'].astype(str).str.lower().eq('open')].copy()

    if df.empty:
        total = pd.DataFrame([{'net_delta': 0.0, 'net_gamma': 0.0, 'net_vega': 0.0, 'net_theta': 0.0}])
        by_expiry = pd.DataFrame()
        by_type = pd.DataFrame()
    else:
        qty = _num(df, 'quantity')
        for greek in ['delta', 'gamma', 'vega', 'theta']:
            df[f'position_{greek}'] = _num(df, greek) * qty
        total = pd.DataFrame([{
            'net_delta': float(df['position_delta'].sum()),
            'net_gamma': float(df['position_gamma'].sum()),
            'net_vega': float(df['position_vega'].sum()),
            'net_theta': float(df['position_theta'].sum()),
        }])
        cols = ['position_delta', 'position_gamma', 'position_vega', 'position_theta']
        by_expiry = df.groupby('expiry', as_index=False)[cols].sum() if 'expiry' in df.columns else pd.DataFrame()
        by_type = df.groupby('option_type', as_index=False)[cols].sum() if 'option_type' in df.columns else pd.DataFrame()

    Path('outputs').mkdir(exist_ok=True)
    total.to_csv('outputs/portfolio_greeks.csv', index=False)
    by_expiry.to_csv('outputs/portfolio_greeks_by_expiry.csv', index=False)
    by_type.to_csv('outputs/portfolio_greeks_by_type.csv', index=False)
    print('Portfolio Greeks complete.')
    return total


if __name__ == '__main__':
    run_portfolio_greeks()
