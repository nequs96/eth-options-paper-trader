from __future__ import annotations
from pathlib import Path
import re
import pandas as pd

REQUIRED_COLUMNS = [
    'timestamp','instrument_name','option_type','expiry','strike','underlying_price_usd',
    'bid_price_usd','ask_price_usd','mark_price_usd','market_price_usd','implied_volatility',
    'delta','gamma','vega','theta','open_interest','volume'
]

ALIASES = {
    'date':'timestamp','time':'timestamp','local_timestamp':'timestamp','ts':'timestamp',
    'instrument':'instrument_name','symbol':'instrument_name','underlying_price':'underlying_price_usd',
    'index_price':'underlying_price_usd','bid':'bid_price_usd','ask':'ask_price_usd','mark_price':'mark_price_usd',
    'price':'market_price_usd','iv':'implied_volatility','implied_vol':'implied_volatility','mark_iv':'implied_volatility',
    'oi':'open_interest','openInterest':'open_interest','vol':'volume'
}

def parse_deribit_instrument(name: str) -> dict:
    # ETH-29MAY26-2500-C
    m=re.match(r'^(?P<underlying>[A-Z]+)-(?P<expiry>\d{1,2}[A-Z]{3}\d{2})-(?P<strike>[0-9.]+)-(?P<cp>[CP])$', str(name))
    if not m: return {}
    return {'expiry':m.group('expiry'),'strike':float(m.group('strike')),'option_type':'call' if m.group('cp')=='C' else 'put'}

def normalize_options_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame(columns=REQUIRED_COLUMNS)
    out=df.copy()
    out.rename(columns={c:ALIASES.get(c,c) for c in out.columns}, inplace=True)
    if 'instrument_name' in out.columns:
        parsed=out['instrument_name'].apply(parse_deribit_instrument)
        for col in ['expiry','strike','option_type']:
            if col not in out.columns:
                out[col]=parsed.apply(lambda x: x.get(col))
    if 'timestamp' in out.columns:
        out['timestamp']=pd.to_datetime(out['timestamp'], errors='coerce', utc=True)
    for c in ['strike','underlying_price_usd','bid_price_usd','ask_price_usd','mark_price_usd','market_price_usd','implied_volatility','delta','gamma','vega','theta','open_interest','volume']:
        if c in out.columns: out[c]=pd.to_numeric(out[c], errors='coerce')
    if 'market_price_usd' not in out.columns:
        bid=out.get('bid_price_usd', pd.Series(index=out.index, dtype=float))
        ask=out.get('ask_price_usd', pd.Series(index=out.index, dtype=float))
        mark=out.get('mark_price_usd', pd.Series(index=out.index, dtype=float))
        out['market_price_usd']=((bid+ask)/2).fillna(mark)
    for col in REQUIRED_COLUMNS:
        if col not in out.columns: out[col]=pd.NA
    out['bid_ask_spread_pct']=((out['ask_price_usd']-out['bid_price_usd'])/out['market_price_usd']).replace([float('inf'),-float('inf')], pd.NA)
    out['moneyness']=(out['strike']-out['underlying_price_usd'])/out['underlying_price_usd']
    out['abs_moneyness']=out['moneyness'].abs()
    return out[REQUIRED_COLUMNS + [c for c in ['bid_ask_spread_pct','moneyness','abs_moneyness'] if c in out.columns]]

def validate_normalized_file(path: str) -> dict:
    p=Path(path)
    if not p.exists() or p.stat().st_size==0: return {'ok':False,'reason':'missing_or_empty'}
    df=pd.read_csv(p)
    missing=[c for c in REQUIRED_COLUMNS if c not in df.columns]
    return {'ok':not missing,'missing_columns':missing,'rows':len(df)}
