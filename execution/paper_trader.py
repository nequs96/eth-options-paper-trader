from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import math
import pandas as pd
from models.market_confidence import MarketConfidence, clean_string
from execution.dynamic_risk_sizing import DynamicRiskConfig, calculate_open_risk_pct, size_position
from execution.dynamic_exit_config import build_dynamic_exit_plan, exit_plan_columns


@dataclass
class PaperTraderConfig:
    candidates_file: str = 'outputs/live_backtest_candidates_filtered.csv'
    positions_file: str = 'outputs/paper_open_positions.csv'
    trade_history_file: str = 'outputs/paper_trade_history.csv'
    initial_cash_file: str = 'outputs/paper_cash.csv'
    initial_cash: float = 10000.0
    min_risk_per_trade: float = 0.001
    normal_max_risk_per_trade: float = 0.0125
    exceptional_max_risk_per_trade: float = 0.02
    max_total_open_risk_pct: float = 0.10
    min_market_price_usd: float = 5.0
    only_trade_cheap_options: bool = True
    min_abs_mispricing_score: float = 0.05
    min_mci_to_trade: float = 0.35
    max_positions: int = 8
    target_positions: int = 3
    max_new_positions_per_cycle: int = 2
    normal_min_score: float = 0.30
    expansion_min_score: float = 0.50
    exceptional_min_score: float = 0.70
    min_relative_to_best_score: float = 0.80
    min_cash_buffer_pct: float = 0.05


def ensure_parent_folder(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def safe_float(value):
    try:
        result = float(value)
    except Exception:
        return None
    return result if math.isfinite(result) else None


def load_csv_if_exists(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists() or file_path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def save_cash(cash: float, config: PaperTraderConfig) -> None:
    ensure_parent_folder(config.initial_cash_file)
    pd.DataFrame([{'cash': float(cash)}]).to_csv(config.initial_cash_file, index=False)


def load_cash(config: PaperTraderConfig) -> float:
    data = load_csv_if_exists(config.initial_cash_file)
    if data.empty or 'cash' not in data.columns:
        save_cash(config.initial_cash, config)
        return float(config.initial_cash)
    values = pd.to_numeric(data['cash'], errors='coerce').dropna()
    return float(values.iloc[-1]) if not values.empty else float(config.initial_cash)


def load_candidates(config: PaperTraderConfig) -> pd.DataFrame:
    return load_csv_if_exists(config.candidates_file)


def open_only(data: pd.DataFrame) -> pd.DataFrame:
    return data if data.empty or 'status' not in data.columns else data[data['status'].astype(str).str.lower().eq('open')]


def load_open_positions(config: PaperTraderConfig) -> pd.DataFrame:
    return open_only(load_csv_if_exists(config.positions_file)).reset_index(drop=True)


def save_open_positions(data: pd.DataFrame, config: PaperTraderConfig) -> None:
    ensure_parent_folder(config.positions_file)
    data.to_csv(config.positions_file, index=False)


def append_trade_history(trades: pd.DataFrame, config: PaperTraderConfig) -> None:
    if trades.empty:
        return
    old = load_csv_if_exists(config.trade_history_file)
    ensure_parent_folder(config.trade_history_file)
    (trades if old.empty else pd.concat([old, trades], ignore_index=True)).to_csv(config.trade_history_file, index=False)


def numeric_series(data: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(data[column], errors='coerce').fillna(default) if column in data.columns else pd.Series([default] * len(data), index=data.index)


def calculate_open_risk_amount(data: pd.DataFrame) -> float:
    return 0.0 if data.empty else float(numeric_series(open_only(data), 'capital_at_risk', 0.0).sum())


def calculate_open_position_value(data: pd.DataFrame) -> float:
    data = open_only(data)
    if data.empty:
        return 0.0
    values = numeric_series(data, 'current_value_usd', 0.0)
    return float(values.sum()) if values.sum() > 0 else float((numeric_series(data, 'current_price_usd', 0.0) * numeric_series(data, 'quantity', 0.0)).sum())


def calculate_unrealized_pnl_amount(data: pd.DataFrame) -> float:
    data = open_only(data)
    if data.empty:
        return 0.0
    return float(((numeric_series(data, 'current_price_usd', 0.0) - numeric_series(data, 'entry_price_usd', 0.0)) * numeric_series(data, 'quantity', 0.0)).sum())


def extract_expiry_from_instrument(value) -> str:
    parts = str(value).split('-')
    return parts[1] if len(parts) >= 2 else ''


def normalize_candidates(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    for column in ['market_price_usd', 'entry_price_usd', 'combined_score', 'mispricing_score', 'price_diff_pct', 'volatility_spread', 'days_to_expiry', 'strike', 'mci', 'edge_score', 'regime_score', 'vol_score', 'liquidity_score', 'greek_score', 'portfolio_score']:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors='coerce')
    if 'classification' in data.columns:
        data['classification'] = data['classification'].astype(str).str.lower().str.strip()
    if 'expiry' not in data.columns and 'instrument_name' in data.columns:
        data['expiry'] = data['instrument_name'].apply(extract_expiry_from_instrument)
    if 'market_price_usd' not in data.columns and 'entry_price_usd' in data.columns:
        data['market_price_usd'] = data['entry_price_usd']
    return data


def get_candidate_score(row) -> float:
    for column in ['mci', 'institutional_edge_score', 'combined_score', 'mispricing_score']:
        if column in row.index:
            value = safe_float(row.get(column))
            if value is not None:
                return abs(value)
    return 0.0


def filter_trade_candidates(candidates: pd.DataFrame, open_positions: pd.DataFrame, config: PaperTraderConfig) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    data = normalize_candidates(candidates)
    if 'instrument_name' in data.columns and 'instrument_name' in open_positions.columns:
        data = data[~data['instrument_name'].astype(str).isin(set(open_positions['instrument_name'].astype(str)))]
    if config.only_trade_cheap_options and 'classification' in data.columns:
        data = data[data['classification'].eq('cheap')]
    if 'mci_reject_reason' in data.columns:
        data = data[data['mci_reject_reason'].apply(clean_string).eq('')]
    if 'market_price_usd' in data.columns:
        data = data[data['market_price_usd'] >= config.min_market_price_usd]
    if data.empty:
        return pd.DataFrame()
    data['portfolio_candidate_score'] = data.apply(get_candidate_score, axis=1)
    return data.sort_values('portfolio_candidate_score', ascending=False).head(config.max_new_positions_per_cycle).reset_index(drop=True)


def market_confidence_from_row(row) -> MarketConfidence:
    return MarketConfidence(safe_float(row.get('mci')) or safe_float(row.get('institutional_edge_score')) or 0.0, safe_float(row.get('edge_score')) or 0.0, safe_float(row.get('regime_score')) or 0.0, safe_float(row.get('vol_score')) or 0.0, safe_float(row.get('liquidity_score')) or 0.5, safe_float(row.get('greek_score')) or 0.0, safe_float(row.get('portfolio_score')) or 1.0, 0.0, 0.0, 0.0, clean_string(row.get('mci_reject_reason', '')))


def open_paper_trades(config: PaperTraderConfig | None = None, current_drawdown: float = 0.0) -> pd.DataFrame:
    config = config or PaperTraderConfig()
    candidates = load_candidates(config)
    open_positions = load_open_positions(config)
    cash = load_cash(config)
    selected = filter_trade_candidates(candidates, open_positions, config)
    if selected.empty:
        print('No candidates passed dynamic portfolio allocation.')
        return pd.DataFrame()
    new_rows = []
    risk_config = DynamicRiskConfig(config.min_risk_per_trade, config.normal_max_risk_per_trade, config.exceptional_max_risk_per_trade, config.max_total_open_risk_pct, config.min_mci_to_trade, config.min_cash_buffer_pct)
    open_risk = calculate_open_risk_pct(open_positions, config.initial_cash)
    now = pd.Timestamp.utcnow().isoformat()
    for _, row in selected.iterrows():
        price = safe_float(row.get('market_price_usd')) or 0.0
        confidence = market_confidence_from_row(row)
        decision = size_position(cash, config.initial_cash, price, confidence, current_drawdown, open_risk, risk_config)
        if not decision.allowed:
            print(f"Skipped {row.get('instrument_name', '')}: {decision.reason}")
            continue
        plan = build_dynamic_exit_plan(confidence, safe_float(row.get('days_to_expiry')) or 0.0)
        position = row.to_dict()
        position.update({'opened_at': now, 'updated_at': now, 'status': 'open', 'entry_price_usd': price, 'current_price_usd': price, 'highest_price_usd': price, 'quantity': decision.quantity, 'capital_at_risk': decision.risk_amount_usd, 'current_value_usd': decision.risk_amount_usd, 'unrealized_pnl_usd': 0.0, 'unrealized_pnl_pct': 0.0, 'confidence_bucket': decision.confidence_bucket, 'dynamic_risk_pct': decision.risk_pct, 'hybrid_exit_reason': 'new_position', **exit_plan_columns(plan)})
        cash -= decision.risk_amount_usd
        new_rows.append(position)
    if not new_rows:
        print('Candidates were selected, but no positions could be sized/opened.')
        return pd.DataFrame()
    opened = pd.DataFrame(new_rows)
    save_open_positions(pd.concat([open_positions, opened], ignore_index=True), config)
    save_cash(cash, config)
    print(f'Opened {len(opened)} new paper position(s).')
    return opened


def print_open_positions_table(config: PaperTraderConfig | None = None, max_rows: int = 50) -> None:
    config = config or PaperTraderConfig()
    data = load_open_positions(config)
    print('\n========== OPEN PAPER POSITIONS =========')
    print('No open paper positions.' if data.empty else data.head(max_rows).to_string(index=False))


def print_paper_account_summary(config: PaperTraderConfig | None = None) -> None:
    config = config or PaperTraderConfig()
    cash = load_cash(config)
    positions = load_open_positions(config)
    value = calculate_open_position_value(positions)
    print('\n========== PAPER ACCOUNT SUMMARY =========')
    print(f'Free cash: ${cash:,.2f}')
    print(f'Open position value: ${value:,.2f}')
    print(f'Total equity: ${cash + value:,.2f}')
    print(f'Open positions: {len(positions)}')
    print('==========================================')
