from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import json
import math
import time
import urllib.parse
import urllib.request
import pandas as pd


@dataclass
class DeribitConfig:
    base_url: str = 'https://www.deribit.com/api/v2'
    currency: str = 'ETH'
    kind: str = 'option'
    expired: bool = False
    timeout: int = 20


def _get_json(url: str, timeout: int = 20) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={'User-Agent': 'eth-options-paper-trader/1.0'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode('utf-8'))


def fetch_deribit_instruments(config: DeribitConfig | None = None) -> list[dict[str, Any]]:
    config = config or DeribitConfig()
    query = urllib.parse.urlencode({'currency': config.currency, 'kind': config.kind, 'expired': str(config.expired).lower()})
    return _get_json(f'{config.base_url}/public/get_instruments?{query}', config.timeout).get('result', [])


def fetch_deribit_ticker(instrument_name: str, config: DeribitConfig | None = None) -> dict[str, Any]:
    config = config or DeribitConfig()
    query = urllib.parse.urlencode({'instrument_name': instrument_name})
    return _get_json(f'{config.base_url}/public/ticker?{query}', config.timeout)


def fetch_eth_underlying_price(config: DeribitConfig | None = None) -> float:
    config = config or DeribitConfig()
    payload = _get_json(f'{config.base_url}/public/get_index_price?index_name=eth_usd', config.timeout)
    return float(payload.get('result', {}).get('index_price', 0.0))


def _option_price_to_usd(value: Any, underlying_price_usd: float) -> float | None:
    try:
        price = float(value)
    except Exception:
        return None
    if not math.isfinite(price) or price <= 0:
        return None
    return float(price * underlying_price_usd) if price < 10 else float(price)


def _option_type_from_instrument(instrument_name: str) -> str:
    suffix = str(instrument_name).split('-')[-1].upper()
    return 'call' if suffix == 'C' else 'put' if suffix == 'P' else ''


def _instrument_expiry_days(instrument: dict[str, Any], now: pd.Timestamp) -> float | None:
    expiry_timestamp = instrument.get('expiration_timestamp')
    if not expiry_timestamp:
        return None
    expiry = pd.to_datetime(expiry_timestamp, unit='ms', utc=True)
    return max((expiry - now).total_seconds() / 86400.0, 0.0)


def _prefilter_and_rank_instruments(instruments: list[dict[str, Any]], underlying_price_usd: float, min_days_to_expiry: float | None, max_days_to_expiry: float | None, max_abs_moneyness: float | None) -> list[dict[str, Any]]:
    now = pd.Timestamp.utcnow()
    ranked: list[tuple[float, float, str, dict[str, Any]]] = []
    for instrument in instruments:
        name = str(instrument.get('instrument_name', ''))
        try:
            strike = float(instrument.get('strike'))
        except Exception:
            continue
        dte = _instrument_expiry_days(instrument, now)
        if not name or strike <= 0 or dte is None:
            continue
        if min_days_to_expiry is not None and dte < min_days_to_expiry:
            continue
        if max_days_to_expiry is not None and dte > max_days_to_expiry:
            continue
        moneyness = abs(strike / underlying_price_usd - 1.0) if underlying_price_usd > 0 else 999.0
        if max_abs_moneyness is not None and moneyness > max_abs_moneyness:
            continue
        ranked.append((dte, moneyness, name, instrument))
    ranked.sort(key=lambda item: (item[0], item[1], item[2]))
    return [item[3] for item in ranked]


def build_live_eth_option_chain(config: DeribitConfig | None = None, max_instruments: int | None = None, progress_every: int = 25, save_partial_path: str | None = None, min_days_to_expiry: float | None = 3.0, max_days_to_expiry: float | None = 60.0, max_abs_moneyness: float | None = 0.35) -> pd.DataFrame:
    config = config or DeribitConfig()
    print('Fetching Deribit ETH option instrument list...')
    instruments = fetch_deribit_instruments(config)
    underlying = fetch_eth_underlying_price(config)
    filtered = _prefilter_and_rank_instruments(instruments, underlying, min_days_to_expiry, max_days_to_expiry, max_abs_moneyness)
    if max_instruments is not None and max_instruments > 0:
        filtered = filtered[:max_instruments]
    total = len(filtered)
    print(f'Deribit instruments available: {len(instruments)}')
    print(f'Underlying ETH index price: ${underlying:,.2f}')
    print(f'After DTE/moneyness prefilter: {len(filtered)}')
    print(f'Processing instruments this run: {total}')
    if total == 0:
        return pd.DataFrame()
    rows = []
    failures = 0
    now = pd.Timestamp.utcnow()
    started = time.time()
    for index, inst in enumerate(filtered, start=1):
        name = inst.get('instrument_name')
        try:
            ticker = fetch_deribit_ticker(name, config).get('result', {})
        except Exception as error:
            ticker = {}
            failures += 1
            if progress_every and (index == 1 or index % progress_every == 0):
                print(f'Warning: ticker failed for {name}: {error}')
        expiry_ts = inst.get('expiration_timestamp')
        expiry = pd.to_datetime(expiry_ts, unit='ms', utc=True) if expiry_ts else pd.NaT
        dte = max((expiry - now).total_seconds() / 86400.0, 0.0) if pd.notna(expiry) else None
        bid = _option_price_to_usd(ticker.get('best_bid_price'), underlying)
        ask = _option_price_to_usd(ticker.get('best_ask_price'), underlying)
        mark = _option_price_to_usd(ticker.get('mark_price'), underlying)
        last = _option_price_to_usd(ticker.get('last_price'), underlying)
        market = mark or last or ((bid + ask) / 2.0 if bid and ask else None)
        spread = (ask - bid) / ((ask + bid) / 2.0) if bid and ask and (ask + bid) > 0 else None
        greeks = ticker.get('greeks') or {}
        mark_iv = ticker.get('mark_iv')
        strike = float(inst.get('strike') or 0.0)
        moneyness = strike / underlying - 1.0 if underlying else None
        rows.append({'instrument_name': name, 'option_type': _option_type_from_instrument(name), 'expiry': expiry.strftime('%Y-%m-%d') if pd.notna(expiry) else '', 'strike': strike, 'days_to_expiry': dte, 'underlying_price_usd': underlying, 'spot_price': underlying, 'moneyness': moneyness, 'abs_moneyness': abs(moneyness) if moneyness is not None else None, 'mark_price_usd': mark, 'bid_price_usd': bid, 'ask_price_usd': ask, 'last_price_usd': last, 'market_price_usd': market, 'mark_iv': mark_iv, 'implied_volatility': float(mark_iv) / 100.0 if mark_iv is not None else None, 'delta': greeks.get('delta'), 'gamma': greeks.get('gamma'), 'vega': greeks.get('vega'), 'theta': greeks.get('theta'), 'open_interest': ticker.get('open_interest'), 'volume': (ticker.get('stats') or {}).get('volume'), 'bid_ask_spread_pct': spread})
        if progress_every and (index % progress_every == 0 or index == total):
            elapsed = max(time.time() - started, 0.001)
            eta = (total - index) / (index / elapsed)
            print(f'Option-chain progress: {index}/{total} ({index / total * 100:5.1f}%) | failures={failures} | ETA={eta:,.0f}s')
        if save_partial_path and progress_every and index % max(progress_every * 4, 1) == 0:
            pd.DataFrame(rows).to_csv(save_partial_path, index=False)
            print(f'Partial option chain saved: {save_partial_path}')
    result = pd.DataFrame(rows)
    print(f'Finished option-chain refresh. Rows={len(result)}, failures={failures}')
    return result
