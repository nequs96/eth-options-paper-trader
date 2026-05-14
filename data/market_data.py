from __future__ import annotations
import json
import urllib.parse
import urllib.request
import pandas as pd


def download_eth_data(start_date: str = '2023-01-01', end_date: str | None = None, interval: str = '1d') -> pd.DataFrame:
    query = urllib.parse.urlencode({'range': '3y', 'interval': interval})
    url = f'https://query1.finance.yahoo.com/v8/finance/chart/ETH-USD?{query}'
    request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 eth-options-paper-trader/1.0'})
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.loads(response.read().decode('utf-8'))
    result = payload.get('chart', {}).get('result', [{}])[0]
    timestamps = result.get('timestamp', [])
    quote = (result.get('indicators', {}).get('quote') or [{}])[0]
    if not timestamps or not quote:
        return pd.DataFrame()
    data = pd.DataFrame(quote)
    data['timestamp'] = pd.to_datetime(timestamps, unit='s', utc=True)
    data = data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
    return data[['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']]


def get_close_prices(data: pd.DataFrame) -> pd.Series:
    if data.empty:
        return pd.Series(dtype=float)
    column = 'Close' if 'Close' in data.columns else 'close'
    return pd.to_numeric(data[column], errors='coerce')


def get_latest_eth_price() -> float:
    closes = get_close_prices(download_eth_data()).dropna()
    return float(closes.iloc[-1]) if not closes.empty else 0.0
