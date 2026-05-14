from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


def fit_simple_surface_residuals(surface: pd.DataFrame) -> pd.DataFrame:
    if surface is None or surface.empty or 'implied_volatility' not in surface.columns:
        return pd.DataFrame()
    df = surface.copy()
    for col in ['implied_volatility','moneyness','days_to_expiry']:
        df[col] = pd.to_numeric(df.get(col), errors='coerce')
    df = df.dropna(subset=['implied_volatility','moneyness','days_to_expiry'])
    if len(df) < 6:
        df['fitted_iv'] = df['implied_volatility']
        df['surface_residual'] = 0.0
        df['surface_residual_zscore'] = 0.0
        return df
    x = df['moneyness'].to_numpy()
    t = np.log1p(df['days_to_expiry'].clip(lower=0).to_numpy())
    y = df['implied_volatility'].to_numpy()
    X = np.column_stack([np.ones(len(df)), x, x * x, t, t * t, x * t])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    residual = y - fitted
    std = residual.std() or 1.0
    df['fitted_iv'] = fitted
    df['surface_residual'] = residual
    df['surface_residual_zscore'] = residual / std
    return df


def run_surface_fit(input_file: str = 'outputs/iv_surface_diagnostics.csv', output_file: str = 'outputs/surface_residuals.csv') -> pd.DataFrame:
    path = Path(input_file)
    df = pd.read_csv(path) if path.exists() and path.stat().st_size > 0 else pd.DataFrame()
    output = fit_simple_surface_residuals(df)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_file, index=False)
    print(f'Saved surface residuals to: {output_file} rows={len(output)}')
    return output


if __name__ == '__main__':
    run_surface_fit()
