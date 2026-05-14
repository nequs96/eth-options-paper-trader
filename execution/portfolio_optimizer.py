from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import math
import pandas as pd


@dataclass(frozen=True)
class OptimizerRiskGateConfig:
    candidates_file: str = 'outputs/live_backtest_candidates_surface_scored.csv'
    scenario_file: str = 'outputs/scenario_portfolio_pnl.csv'
    greeks_file: str = 'outputs/portfolio_greeks.csv'
    output_file: str = 'outputs/optimized_trade_list.csv'
    rejected_file: str = 'outputs/optimizer_rejected_candidates.csv'
    report_file: str = 'outputs/optimizer_risk_gate_report.csv'
    max_positions: int = 2
    min_institutional_edge_score: float = 0.45
    min_days_to_expiry: float = 7.0
    max_days_to_expiry: float = 35.0
    max_abs_moneyness: float = 0.12
    min_open_interest: float = 100.0
    min_volume: float = 20.0
    max_spread_front_week: float = 0.06
    max_spread_front_month: float = 0.08
    max_spread_other: float = 0.10
    max_abs_net_delta_after_hint: float = 5.0
    max_abs_net_vega_after_hint: float = 60.0
    max_abs_net_theta_after_hint: float = 50.0
    max_flat_7d_loss_abs: float = 350.0
    require_classification_cheap: bool = True
    allow_front_week_exception: bool = False
    exceptional_front_week_min_score: float = 0.60
    exceptional_front_week_min_oi: float = 500.0
    exceptional_front_week_min_volume: float = 100.0


def _load_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors='coerce').fillna(default)


def _safe_float(value, default: float = 0.0) -> float:
    try:
        result = float(value)
    except Exception:
        return default
    return result if math.isfinite(result) else default


def fractional_kelly(edge, win_prob: float = 0.55, payoff_ratio: float = 1.5, fraction: float = 0.25) -> float:
    b = max(float(payoff_ratio), 1e-9)
    p = float(win_prob)
    kelly = (b * p - (1.0 - p)) / b
    return max(kelly, 0.0) * float(fraction) * max(float(edge), 0.0)


def _term_bucket_from_dte(dte: float) -> str:
    if dte <= 7:
        return 'front_week'
    if dte <= 35:
        return 'front_month'
    return 'other'


def _spread_limit(row: pd.Series, cfg: OptimizerRiskGateConfig) -> float:
    bucket = str(row.get('term_bucket') or _term_bucket_from_dte(_safe_float(row.get('days_to_expiry')))).lower()
    if bucket == 'front_week':
        return cfg.max_spread_front_week
    if bucket == 'front_month':
        return cfg.max_spread_front_month
    return cfg.max_spread_other


def _flat_7d_loss(scenarios: pd.DataFrame) -> float:
    if scenarios.empty or 'portfolio_scenario_pnl' not in scenarios.columns:
        return 0.0
    data = scenarios.copy()
    for col in ['spot_shock', 'iv_shock', 'days_forward', 'portfolio_scenario_pnl']:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    target = data[(data.get('spot_shock', 999) == 0.0) & (data.get('iv_shock', 999) == 0.0) & (data.get('days_forward', 999) == 7)]
    if target.empty:
        return 0.0
    return min(float(target['portfolio_scenario_pnl'].iloc[-1]), 0.0)


def _current_greeks(greeks: pd.DataFrame) -> dict[str, float]:
    if greeks.empty:
        return {'net_delta': 0.0, 'net_gamma': 0.0, 'net_vega': 0.0, 'net_theta': 0.0}
    row = greeks.iloc[-1]
    return {
        'net_delta': _safe_float(row.get('net_delta')),
        'net_gamma': _safe_float(row.get('net_gamma')),
        'net_vega': _safe_float(row.get('net_vega')),
        'net_theta': _safe_float(row.get('net_theta')),
    }


def _candidate_risk_adjusted_score(data: pd.DataFrame, cfg: OptimizerRiskGateConfig) -> pd.Series:
    edge = _num(data, 'institutional_edge_score', _num(data, 'combined_score', 0.0)).clip(0, 1)
    dte = _num(data, 'days_to_expiry', 0.0)
    spread = _num(data, 'bid_ask_spread_pct', 0.0).clip(lower=0)
    abs_moneyness = _num(data, 'abs_moneyness', _num(data, 'moneyness', 0.0).abs()).clip(lower=0)
    oi = _num(data, 'open_interest', 0.0).clip(lower=0)
    volume = _num(data, 'volume', 0.0).clip(lower=0)

    dte_score = ((dte - cfg.min_days_to_expiry) / max(cfg.max_days_to_expiry - cfg.min_days_to_expiry, 1.0)).clip(0, 1)
    # Prefer middle of the allowed window rather than very short or very long.
    dte_score = 1.0 - (dte_score - 0.45).abs() / 0.55
    dte_score = dte_score.clip(0, 1)
    spread_score = (1.0 - spread / 0.10).clip(0, 1)
    moneyness_score = (1.0 - abs_moneyness / cfg.max_abs_moneyness).clip(0, 1)
    oi_score = (oi / max(cfg.min_open_interest * 5, 1)).clip(0, 1)
    volume_score = (volume / max(cfg.min_volume * 5, 1)).clip(0, 1)

    return (
        0.40 * edge
        + 0.18 * dte_score
        + 0.15 * spread_score
        + 0.12 * moneyness_score
        + 0.08 * oi_score
        + 0.07 * volume_score
    ).clip(0, 1)


def _rejection_reasons(row: pd.Series, cfg: OptimizerRiskGateConfig) -> list[str]:
    reasons: list[str] = []
    score = _safe_float(row.get('institutional_edge_score', row.get('combined_score', 0.0)))
    dte = _safe_float(row.get('days_to_expiry'))
    spread = _safe_float(row.get('bid_ask_spread_pct'))
    abs_m = abs(_safe_float(row.get('abs_moneyness', row.get('moneyness', 0.0))))
    oi = _safe_float(row.get('open_interest'))
    volume = _safe_float(row.get('volume'))
    classification = str(row.get('classification', '')).lower().strip()
    spread_limit = _spread_limit(row, cfg)

    is_front_week = dte < cfg.min_days_to_expiry
    front_week_exception = (
        cfg.allow_front_week_exception
        and is_front_week
        and score >= cfg.exceptional_front_week_min_score
        and oi >= cfg.exceptional_front_week_min_oi
        and volume >= cfg.exceptional_front_week_min_volume
        and spread <= cfg.max_spread_front_week
    )

    if cfg.require_classification_cheap and classification and classification != 'cheap':
        reasons.append('classification_not_cheap')
    if score < cfg.min_institutional_edge_score:
        reasons.append('edge_score_below_optimizer_minimum')
    if dte < cfg.min_days_to_expiry and not front_week_exception:
        reasons.append('dte_below_optimizer_minimum')
    if dte > cfg.max_days_to_expiry:
        reasons.append('dte_above_optimizer_maximum')
    if abs_m > cfg.max_abs_moneyness:
        reasons.append('abs_moneyness_too_high')
    if spread > spread_limit:
        reasons.append('spread_above_dte_bucket_limit')
    if oi < cfg.min_open_interest:
        reasons.append('open_interest_below_minimum')
    if volume < cfg.min_volume:
        reasons.append('volume_below_minimum')
    return reasons


def optimize_trade_list(
    candidates_file: str = 'outputs/live_backtest_candidates_surface_scored.csv',
    max_positions: int = 2,
    config: OptimizerRiskGateConfig | None = None,
) -> pd.DataFrame:
    cfg = config or OptimizerRiskGateConfig(candidates_file=candidates_file, max_positions=max_positions)
    candidates = _load_csv(cfg.candidates_file)
    scenarios = _load_csv(cfg.scenario_file)
    greeks = _load_csv(cfg.greeks_file)
    Path('outputs').mkdir(exist_ok=True)

    if candidates.empty:
        empty = pd.DataFrame()
        empty.to_csv(cfg.output_file, index=False)
        empty.to_csv(cfg.rejected_file, index=False)
        pd.DataFrame([{**asdict(cfg), 'status': 'no_candidates'}]).to_csv(cfg.report_file, index=False)
        print('Portfolio optimizer complete. selected=0 rejected=0 status=no_candidates')
        return empty

    data = candidates.copy()
    for col in ['institutional_edge_score', 'combined_score', 'days_to_expiry', 'bid_ask_spread_pct', 'abs_moneyness', 'moneyness', 'open_interest', 'volume', 'delta', 'vega', 'theta']:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

    data['optimizer_risk_adjusted_score'] = _candidate_risk_adjusted_score(data, cfg)
    data['optimizer_reject_reason'] = data.apply(lambda row: ';'.join(_rejection_reasons(row, cfg)), axis=1)
    accepted = data[data['optimizer_reject_reason'].astype(str).eq('')].copy()
    rejected = data[~data['optimizer_reject_reason'].astype(str).eq('')].copy()

    current_greeks = _current_greeks(greeks)
    flat_7d_loss = _flat_7d_loss(scenarios)
    portfolio_gate_reasons = []
    if abs(current_greeks['net_theta']) > cfg.max_abs_net_theta_after_hint:
        portfolio_gate_reasons.append('current_portfolio_theta_budget_exceeded')
    if abs(current_greeks['net_delta']) > cfg.max_abs_net_delta_after_hint:
        portfolio_gate_reasons.append('current_portfolio_delta_budget_exceeded')
    if abs(current_greeks['net_vega']) > cfg.max_abs_net_vega_after_hint:
        portfolio_gate_reasons.append('current_portfolio_vega_budget_exceeded')
    if abs(flat_7d_loss) > cfg.max_flat_7d_loss_abs:
        portfolio_gate_reasons.append('flat_7d_scenario_loss_budget_exceeded')

    # If current portfolio is already beyond hard limits, do not add new positions.
    if portfolio_gate_reasons:
        if not accepted.empty:
            temp = accepted.copy()
            temp['optimizer_reject_reason'] = ';'.join(portfolio_gate_reasons)
            rejected = pd.concat([rejected, temp], ignore_index=True)
        selected = pd.DataFrame(columns=accepted.columns)
        status = 'portfolio_gate_blocked_new_trades'
    else:
        selected = accepted.sort_values('optimizer_risk_adjusted_score', ascending=False).head(cfg.max_positions).copy()
        status = 'ok'

    if not selected.empty:
        selected['optimizer_reason'] = 'risk_gated_top_risk_adjusted_score'
        score_col = 'optimizer_risk_adjusted_score'
        selected['fractional_kelly_size_hint'] = selected[score_col].apply(fractional_kelly)

    selected.to_csv(cfg.output_file, index=False)
    rejected.to_csv(cfg.rejected_file, index=False)

    report = {
        **asdict(cfg),
        'status': status,
        'input_candidates': len(candidates),
        'candidate_gate_passed': len(accepted),
        'candidate_gate_rejected': len(rejected),
        'selected': len(selected),
        'current_net_delta': current_greeks['net_delta'],
        'current_net_gamma': current_greeks['net_gamma'],
        'current_net_vega': current_greeks['net_vega'],
        'current_net_theta': current_greeks['net_theta'],
        'flat_7d_loss': flat_7d_loss,
        'portfolio_gate_reasons': ';'.join(portfolio_gate_reasons),
    }
    pd.DataFrame([report]).to_csv(cfg.report_file, index=False)
    print(f"Portfolio optimizer complete. selected={len(selected)} rejected={len(rejected)} status={status}")
    if portfolio_gate_reasons:
        print('Portfolio gate reasons: ' + '; '.join(portfolio_gate_reasons))
    return selected


if __name__ == '__main__':
    optimize_trade_list()
