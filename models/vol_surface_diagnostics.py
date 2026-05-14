from __future__ import annotations
from pathlib import Path
import math
import pandas as pd


def _z(series: pd.Series) -> pd.Series:
    series = pd.to_numeric(series, errors='coerce')
    std = series.std(ddof=0)
    if not std or not math.isfinite(float(std)):
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - series.mean()) / std


def _term(dte: float) -> str:
    if dte <= 7: return 'front_week'
    if dte <= 21: return 'front_month'
    if dte <= 60: return 'mid_curve'
    return 'back_curve'


def compute_iv_surface_diagnostics(option_chain: pd.DataFrame) -> pd.DataFrame:
    if option_chain is None or option_chain.empty:
        return pd.DataFrame()
    df = option_chain.copy()
    for c in ['implied_volatility','mark_iv','days_to_expiry','strike','underlying_price_usd','spot_price','bid_ask_spread_pct','market_price_usd']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    if 'implied_volatility' not in df.columns or df['implied_volatility'].isna().all():
        df['implied_volatility'] = df.get('mark_iv', pd.Series([pd.NA] * len(df), index=df.index)) / 100.0
    if 'underlying_price_usd' not in df.columns and 'spot_price' in df.columns:
        df['underlying_price_usd'] = df['spot_price']
    if 'moneyness' not in df.columns:
        df['moneyness'] = df['strike'] / df['underlying_price_usd'] - 1.0
    df['abs_moneyness'] = df['moneyness'].abs()
    df['term_bucket'] = df['days_to_expiry'].apply(lambda x: _term(float(x)) if pd.notna(x) else 'unknown')
    df['surface_iv_zscore'] = _z(df['implied_volatility']).fillna(0.0)
    df['surface_iv_percentile'] = df['implied_volatility'].rank(pct=True).fillna(0.5)
    df['same_expiry_iv_zscore'] = 0.0
    df['same_expiry_iv_percentile'] = 0.5
    if 'expiry' in df.columns:
        for _, idx in df.groupby('expiry').groups.items():
            iv = df.loc[idx, 'implied_volatility']
            df.loc[idx, 'same_expiry_iv_zscore'] = _z(iv).fillna(0.0)
            df.loc[idx, 'same_expiry_iv_percentile'] = iv.rank(pct=True).fillna(0.5)
    spread = pd.to_numeric(df.get('bid_ask_spread_pct', 0.0), errors='coerce').fillna(0.0).clip(0, 1)
    df['surface_relative_value_score'] = (0.35 * (1 - df['same_expiry_iv_percentile']).clip(0, 1) + 0.25 * (-df['surface_iv_zscore']).clip(0, 3) / 3 + 0.20 * (1 - df['abs_moneyness'].clip(0, 1)) + 0.20 * (1 - spread / 0.35).clip(0, 1)).clip(0, 1)
    cols = ['instrument_name','option_type','expiry','strike','days_to_expiry','underlying_price_usd','market_price_usd','implied_volatility','moneyness','abs_moneyness','term_bucket','same_expiry_iv_zscore','same_expiry_iv_percentile','surface_iv_zscore','surface_iv_percentile','surface_relative_value_score','bid_ask_spread_pct','delta','gamma','vega','theta','open_interest','volume']
    return df[[c for c in cols if c in df.columns]]


def run_surface_diagnostics(option_chain_file: str = 'outputs/live_eth_option_chain.csv', output_file: str = 'outputs/iv_surface_diagnostics.csv') -> pd.DataFrame:
    path = Path(option_chain_file)
    chain = pd.read_csv(path) if path.exists() and path.stat().st_size > 0 else pd.DataFrame()
    output = compute_iv_surface_diagnostics(chain)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_file, index=False)
    print(f'Saved IV surface diagnostics to: {output_file} rows={len(output)}')
    return output


if __name__ == '__main__':
    run_surface_diagnostics()
