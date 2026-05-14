from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import urllib.error
import pandas as pd
from data.options_data import DeribitConfig, build_live_eth_option_chain
from data.market_data import download_eth_data, get_close_prices
from models.black_scholes import black_scholes_price
from models.volatility import historical_volatility
from strategies.option_mispricing import classify_option_mispricing
from backtesting.portfolio import Portfolio

DEFAULT_FALLBACK_HISTORICAL_VOLATILITY = 0.75
CANDIDATE_COLUMNS = ['instrument_name','option_type','expiry','strike','days_to_expiry','underlying_price_usd','spot_price','moneyness','abs_moneyness','market_price_usd','bid_ask_spread_pct','implied_volatility','model_price_usd','theoretical_price','price_diff_pct','volatility_spread','classification','combined_score','historical_volatility_used']


@dataclass
class LiveOptionBacktestConfig:
    refresh_option_chain: bool = True
    option_chain_file: str = 'outputs/live_eth_option_chain.csv'
    output_folder: str = 'outputs'
    historical_vol_start_date: str = '2023-01-01'
    initial_cash: float = 10000.0
    max_risk_per_trade: float = 0.01
    max_positions: int = 5
    risk_free_rate: float = 0.04
    min_days_to_expiry: float = 3.0
    max_days_to_expiry: float = 45.0
    min_market_price_usd: float = 5.0
    max_bid_ask_spread_pct: float = 0.35
    price_threshold: float = 0.10
    volatility_threshold: float = 0.10
    min_volatility: float = 0.10
    max_volatility: float = 2.50
    allow_calls: bool = True
    allow_puts: bool = True
    only_trade_cheap_options: bool = True
    max_option_chain_instruments: int | None = 150
    ticker_timeout_seconds: int = 5
    option_chain_progress_every: int = 25
    save_partial_option_chain: bool = True
    fallback_historical_volatility: float = DEFAULT_FALLBACK_HISTORICAL_VOLATILITY
    scanner_max_abs_moneyness: float | None = 0.35


@dataclass
class LiveOptionBacktestResult:
    candidates: pd.DataFrame
    positions: pd.DataFrame
    portfolio: Portfolio


def ensure_output_folder(folder: str) -> None:
    Path(folder).mkdir(parents=True, exist_ok=True)


def get_or_refresh_live_option_chain(config: LiveOptionBacktestConfig) -> pd.DataFrame:
    ensure_output_folder(config.output_folder)
    if not config.refresh_option_chain:
        print(f'Loading cached option chain: {config.option_chain_file}')
        path = Path(config.option_chain_file)
        return pd.read_csv(path) if path.exists() and path.stat().st_size > 0 else pd.DataFrame()
    print('Refreshing live Deribit ETH option chain...')
    partial = str(Path(config.output_folder) / 'live_eth_option_chain.partial.csv') if config.save_partial_option_chain else None
    chain = build_live_eth_option_chain(DeribitConfig(timeout=config.ticker_timeout_seconds), max_instruments=config.max_option_chain_instruments, progress_every=config.option_chain_progress_every, save_partial_path=partial, min_days_to_expiry=config.min_days_to_expiry, max_days_to_expiry=max(config.max_days_to_expiry, 60.0), max_abs_moneyness=config.scanner_max_abs_moneyness)
    chain.to_csv(config.option_chain_file, index=False)
    print(f'Saved live option chain to: {config.option_chain_file}')
    return chain


def estimate_current_historical_volatility(start_date: str, fallback_historical_volatility: float = DEFAULT_FALLBACK_HISTORICAL_VOLATILITY) -> float:
    try:
        close = get_close_prices(download_eth_data(start_date=start_date)).dropna()
        hv = historical_volatility(close, '1d')
        if hv and pd.notna(hv) and hv > 0:
            return max(float(hv), 0.05)
    except urllib.error.HTTPError as error:
        print(f'Warning: HTTP error while downloading ETH history: {error}')
    except Exception as error:
        print(f'Warning: could not estimate ETH historical volatility: {error}')
    print(f'Using fallback historical volatility: {fallback_historical_volatility:.2%}')
    return float(fallback_historical_volatility)


def _safe_float(value, default: float = 0.0) -> float:
    try:
        result = float(value)
    except Exception:
        return default
    return result if pd.notna(result) else default


def _reject(row: pd.Series, reason: str) -> dict:
    result = row.to_dict()
    result['raw_reject_reason'] = reason
    return result


def build_candidate_tables(option_chain: pd.DataFrame, historical_volatility_value: float, config: LiveOptionBacktestConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    accepted, rejected = [], []
    if option_chain.empty:
        return pd.DataFrame(columns=CANDIDATE_COLUMNS), pd.DataFrame(columns=['raw_reject_reason'])
    data = option_chain.copy()
    for column in ['days_to_expiry','market_price_usd','bid_ask_spread_pct','strike','spot_price','underlying_price_usd','implied_volatility','moneyness','abs_moneyness','delta','open_interest','volume']:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors='coerce')
    for _, row in data.iterrows():
        dte = _safe_float(row.get('days_to_expiry'))
        market = _safe_float(row.get('market_price_usd'))
        spread = row.get('bid_ask_spread_pct')
        option_type = str(row.get('option_type', '')).lower().strip()
        spot = _safe_float(row.get('spot_price') or row.get('underlying_price_usd'))
        strike = _safe_float(row.get('strike'))
        if dte < config.min_days_to_expiry:
            rejected.append(_reject(row, 'dte_too_low')); continue
        if dte > config.max_days_to_expiry:
            rejected.append(_reject(row, 'dte_too_high')); continue
        if market <= 0:
            rejected.append(_reject(row, 'missing_or_invalid_market_price')); continue
        if market < config.min_market_price_usd:
            rejected.append(_reject(row, 'market_price_below_minimum')); continue
        if pd.notna(spread) and float(spread) > config.max_bid_ask_spread_pct:
            rejected.append(_reject(row, 'spread_too_wide')); continue
        if option_type == 'call' and not config.allow_calls:
            rejected.append(_reject(row, 'calls_disabled')); continue
        if option_type == 'put' and not config.allow_puts:
            rejected.append(_reject(row, 'puts_disabled')); continue
        if spot <= 0 or strike <= 0:
            rejected.append(_reject(row, 'invalid_spot_or_strike')); continue
        model = black_scholes_price(spot, strike, max(dte / 365.0, 1.0 / 365.0), config.risk_free_rate, historical_volatility_value, option_type)
        price_diff_pct = (market - model) / model if model > 0 else 0.0
        iv = _safe_float(row.get('implied_volatility'), historical_volatility_value)
        vol_spread = iv - historical_volatility_value
        classification = classify_option_mispricing(price_difference_percent=price_diff_pct, vol_spread=vol_spread, price_threshold=config.price_threshold, volatility_threshold=config.volatility_threshold)
        output = row.to_dict() | {'model_price_usd': model, 'theoretical_price': model, 'price_diff_pct': price_diff_pct, 'volatility_spread': vol_spread, 'classification': classification, 'combined_score': abs(price_diff_pct) + abs(vol_spread), 'historical_volatility_used': historical_volatility_value}
        if config.only_trade_cheap_options and classification != 'cheap':
            output['raw_reject_reason'] = f'classification_{classification}'
            rejected.append(output)
        else:
            accepted.append(output)
    acc = pd.DataFrame(accepted)
    rej = pd.DataFrame(rejected)
    if acc.empty:
        acc = pd.DataFrame(columns=CANDIDATE_COLUMNS)
    elif 'combined_score' in acc.columns:
        acc = acc.sort_values('combined_score', ascending=False).reset_index(drop=True)
    if rej.empty:
        rej = pd.DataFrame(columns=['raw_reject_reason'])
    return acc, rej


def build_candidate_table(option_chain: pd.DataFrame, historical_volatility_value: float, config: LiveOptionBacktestConfig) -> pd.DataFrame:
    return build_candidate_tables(option_chain, historical_volatility_value, config)[0]


def print_rejection_summary(rejected: pd.DataFrame) -> None:
    print('\n========== RAW CANDIDATE REJECTION SUMMARY =========')
    if rejected.empty or 'raw_reject_reason' not in rejected.columns:
        print('No rejected rows recorded.')
    else:
        counts = rejected['raw_reject_reason'].astype(str).value_counts().reset_index()
        counts.columns = ['reason', 'count']
        print(counts.to_string(index=False))
    print('====================================================')


def run_live_option_paper_backtest(config: LiveOptionBacktestConfig | None = None) -> LiveOptionBacktestResult:
    config = config or LiveOptionBacktestConfig()
    ensure_output_folder(config.output_folder)
    chain = get_or_refresh_live_option_chain(config)
    print('Downloading ETH historical data for volatility estimate...')
    hv = estimate_current_historical_volatility(config.historical_vol_start_date, config.fallback_historical_volatility)
    print(f'Estimated ETH historical volatility used: {hv:.2%}')
    candidates, rejected = build_candidate_tables(chain, hv, config)
    candidates_file = str(Path(config.output_folder) / 'live_backtest_candidates.csv')
    rejected_file = str(Path(config.output_folder) / 'live_backtest_candidates_raw_rejected.csv')
    candidates.to_csv(candidates_file, index=False)
    rejected.to_csv(rejected_file, index=False)
    print(f'Saved candidates to:      {candidates_file}')
    print(f'Saved raw rejections to:  {rejected_file}')
    print('\n========== LIVE OPTION PAPER-BACKTEST SUMMARY =========')
    print(f'Option-chain rows:        {len(chain)}')
    print(f'Scanned candidates:       {len(candidates)}')
    print(f"Cheap options accepted:   {int((candidates.get('classification', pd.Series(dtype=str)) == 'cheap').sum()) if not candidates.empty else 0}")
    print('=======================================================')
    print_rejection_summary(rejected)
    return LiveOptionBacktestResult(candidates, pd.DataFrame(), Portfolio(config.initial_cash, config.initial_cash))
