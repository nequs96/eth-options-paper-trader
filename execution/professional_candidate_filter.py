from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from data.market_data import download_eth_data, get_close_prices
from models.eth_forward_volatility import calculate_volatility_forecast
from models.trend_regime_model import get_trend_regime
from models.market_confidence import calculate_market_confidence


@dataclass
class ProfessionalFilterConfig:
    raw_candidates_file: str = 'outputs/live_backtest_candidates.csv'
    filtered_candidates_file: str = 'outputs/live_backtest_candidates_filtered.csv'
    rejected_candidates_file: str = 'outputs/live_backtest_candidates_rejected.csv'
    historical_vol_start_date: str = '2023-01-01'
    min_days_to_expiry: float = 3.0
    max_days_to_expiry: float = 45.0
    min_market_price_usd: float = 5.0
    max_bid_ask_spread_pct: float = 0.35
    min_mci_to_accept: float = 0.35
    min_liquidity_score: float = 0.20
    min_edge_score: float = 0.20


def ensure_parent_folder(file_path: str) -> None:
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def load_csv_if_exists(file_path: str) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except Exception as error:
        print(f'Candidate filter: could not read {file_path}: {error}')
        return pd.DataFrame()


def _write_empty_outputs(config: ProfessionalFilterConfig) -> pd.DataFrame:
    ensure_parent_folder(config.filtered_candidates_file)
    ensure_parent_folder(config.rejected_candidates_file)
    pd.DataFrame().to_csv(config.filtered_candidates_file, index=False)
    pd.DataFrame().to_csv(config.rejected_candidates_file, index=False)
    return pd.DataFrame()


def normalize_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    data = candidates.copy()
    for column in ['market_price_usd','model_price_usd','theoretical_price','price_diff_pct','volatility_spread','days_to_expiry','bid_ask_spread_pct','strike','spot_price','underlying_price_usd','delta','gamma','vega','theta','open_interest','implied_volatility','mark_iv','iv','combined_score']:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors='coerce')
    if 'classification' in data.columns:
        data['classification'] = data['classification'].astype(str).str.lower().str.strip()
    if 'option_type' in data.columns:
        data['option_type'] = data['option_type'].astype(str).str.lower().str.strip().replace({'c': 'call', 'p': 'put'})
    return data


def _get_market_context(config: ProfessionalFilterConfig):
    try:
        close = get_close_prices(download_eth_data(start_date=config.historical_vol_start_date)).dropna()
        vol_forecast = calculate_volatility_forecast(close)
    except Exception:
        vol_forecast = calculate_volatility_forecast(pd.Series(dtype=float))
    trend = get_trend_regime(config.historical_vol_start_date)
    return vol_forecast, trend


def filter_candidate_file(config: ProfessionalFilterConfig | None = None) -> pd.DataFrame:
    config = config or ProfessionalFilterConfig()
    raw = load_csv_if_exists(config.raw_candidates_file)
    ensure_parent_folder(config.filtered_candidates_file)
    ensure_parent_folder(config.rejected_candidates_file)
    if raw.empty:
        print('No raw candidates to filter. Writing empty filtered/rejected files and continuing cycle.')
        return _write_empty_outputs(config)
    data = normalize_candidates(raw)
    vol_forecast, trend = _get_market_context(config)
    accepted, rejected = [], []
    for _, row in data.iterrows():
        reject_reason = ''
        if row.get('days_to_expiry', 0) < config.min_days_to_expiry:
            reject_reason = 'dte_too_low'
        elif row.get('days_to_expiry', 999) > config.max_days_to_expiry:
            reject_reason = 'dte_too_high'
        elif row.get('market_price_usd', 0) < config.min_market_price_usd:
            reject_reason = 'market_price_below_minimum'
        elif pd.notna(row.get('bid_ask_spread_pct')) and row.get('bid_ask_spread_pct') > config.max_bid_ask_spread_pct:
            reject_reason = 'spread_too_wide'
        confidence = calculate_market_confidence(row, vol_forecast, trend)
        output = row.to_dict() | {'mci': confidence.mci, 'edge_score': confidence.edge_score, 'regime_score': confidence.regime_score, 'vol_score': confidence.vol_score, 'liquidity_score': confidence.liquidity_score, 'greek_score': confidence.greek_score, 'portfolio_score': confidence.portfolio_score, 'mci_reject_reason': confidence.reject_reason, 'decision_reason': 'accepted_dynamic_mci'}
        if not reject_reason and confidence.reject_reason:
            reject_reason = confidence.reject_reason
        if not reject_reason and confidence.mci < config.min_mci_to_accept:
            reject_reason = 'mci_below_threshold'
        if reject_reason:
            output['decision'] = 'reject'
            output['mci_reject_reason'] = reject_reason
            rejected.append(output)
        else:
            output['decision'] = 'accept'
            accepted.append(output)
    filtered = pd.DataFrame(accepted)
    rejected_df = pd.DataFrame(rejected)
    filtered.to_csv(config.filtered_candidates_file, index=False)
    rejected_df.to_csv(config.rejected_candidates_file, index=False)
    print(f'Professional filter accepted={len(filtered)} rejected={len(rejected_df)}')
    return filtered


if __name__ == '__main__':
    filter_candidate_file()
