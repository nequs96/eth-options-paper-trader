from __future__ import annotations
import pandas as pd


def classify_regime(prices: pd.Series | None = None, hv_fast: float | None = None, hv_slow: float | None = None) -> dict:
    if prices is None or len(prices) < 50:
        return {'regime': 'unknown', 'trend': 'unknown', 'vol_state': 'unknown'}
    s = pd.to_numeric(prices, errors='coerce').dropna()
    if len(s) < 50:
        return {'regime': 'unknown', 'trend': 'unknown', 'vol_state': 'unknown'}
    fast = float(s.tail(20).mean())
    slow = float(s.tail(50).mean())
    trend = 'bull' if fast > slow * 1.005 else 'bear' if fast < slow * 0.995 else 'range'
    vol_state = 'expanding' if hv_fast and hv_slow and hv_fast > hv_slow * 1.05 else 'normal'
    return {'regime': f'{trend}_{vol_state}', 'trend': trend, 'vol_state': vol_state}
