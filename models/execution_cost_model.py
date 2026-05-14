from __future__ import annotations
import pandas as pd


def add_execution_cost_estimates(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates is None or candidates.empty:
        return pd.DataFrame()
    df = candidates.copy()
    for col in ['market_price_usd','bid_price_usd','ask_price_usd','bid_ask_spread_pct']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    mark = df.get('market_price_usd', pd.Series([0.0] * len(df), index=df.index)).fillna(0.0)
    spread = df.get('bid_ask_spread_pct', pd.Series([0.0] * len(df), index=df.index)).fillna(0.0)
    df['estimated_entry_price_usd'] = df.get('ask_price_usd', pd.Series([pd.NA] * len(df), index=df.index)).fillna(mark * (1 + spread / 2))
    df['estimated_exit_price_usd'] = df.get('bid_price_usd', pd.Series([pd.NA] * len(df), index=df.index)).fillna(mark * (1 - spread / 2))
    df['round_trip_spread_cost_usd'] = (df['estimated_entry_price_usd'] - df['estimated_exit_price_usd']).clip(lower=0)
    df['execution_quality_score'] = (1 - spread / 0.35).clip(0, 1)
    return df
