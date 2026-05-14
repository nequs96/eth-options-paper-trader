from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd

@dataclass(frozen=True)
class PortfolioRiskLimitConfig:
    positions_file: str = 'outputs/paper_open_positions.csv'
    greeks_file: str = 'outputs/portfolio_greeks.csv'
    scenario_file: str = 'outputs/scenario_portfolio_pnl.csv'
    report_file: str = 'outputs/portfolio_limit_report.csv'
    breaches_file: str = 'outputs/risk_limit_breaches.csv'
    max_open_positions: int = 8
    max_abs_net_delta: float = 5.0
    max_abs_net_gamma: float = 0.10
    max_abs_net_vega: float = 60.0
    max_abs_net_theta: float = 50.0
    max_flat_7d_loss_abs: float = 350.0
    max_down_10_iv_down_10_loss_abs: float = 500.0
    max_same_expiry_positions: int = 3
    max_same_option_type_positions: int = 5


def _load(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def _sf(v, d: float = 0.0) -> float:
    try:
        x = float(v)
    except Exception:
        return d
    return x if pd.notna(x) else d


def _last(df: pd.DataFrame, col: str) -> float:
    return 0.0 if df.empty or col not in df.columns else _sf(df.iloc[-1].get(col))


def _scenario(df: pd.DataFrame, spot: float, iv: float, days: int) -> float:
    if df.empty or 'portfolio_scenario_pnl' not in df.columns:
        return 0.0
    d = df.copy()
    for c in ['spot_shock', 'iv_shock', 'days_forward', 'portfolio_scenario_pnl']:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors='coerce')
    t = d[(d.get('spot_shock') == spot) & (d.get('iv_shock') == iv) & (d.get('days_forward') == days)]
    return 0.0 if t.empty else _sf(t.iloc[-1].get('portfolio_scenario_pnl'))


def check_portfolio_risk_limits(config: PortfolioRiskLimitConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = config or PortfolioRiskLimitConfig()
    pos, g, s = _load(cfg.positions_file), _load(cfg.greeks_file), _load(cfg.scenario_file)
    Path('outputs').mkdir(exist_ok=True)
    op = pos[pos['status'].astype(str).str.lower().eq('open')].copy() if not pos.empty and 'status' in pos.columns else pos.copy()
    metrics = {
        **asdict(cfg),
        'open_positions': int(len(op)),
        'net_delta': _last(g, 'net_delta'),
        'net_gamma': _last(g, 'net_gamma'),
        'net_vega': _last(g, 'net_vega'),
        'net_theta': _last(g, 'net_theta'),
        'flat_7d_loss': min(_scenario(s, 0.0, 0.0, 7), 0.0),
        'down_10_iv_down_10_7d_loss': min(_scenario(s, -0.1, -0.1, 7), 0.0),
        'max_same_expiry_count': int(op['expiry'].astype(str).value_counts().max()) if not op.empty and 'expiry' in op.columns else 0,
        'max_same_option_type_count': int(op['option_type'].astype(str).value_counts().max()) if not op.empty and 'option_type' in op.columns else 0,
    }
    breaches = []
    def add(name: str, value: float, limit: float, mode: str = 'abs') -> None:
        hit = value > limit if mode == 'gt' else abs(value) > limit
        if hit:
            breaches.append({'limit_name': name, 'value': value, 'limit': limit, 'severity': 'HARD', 'action': 'block_new_trades_or_reduce_risk'})
    add('max_open_positions', metrics['open_positions'], cfg.max_open_positions, 'gt')
    add('max_abs_net_delta', metrics['net_delta'], cfg.max_abs_net_delta)
    add('max_abs_net_gamma', metrics['net_gamma'], cfg.max_abs_net_gamma)
    add('max_abs_net_vega', metrics['net_vega'], cfg.max_abs_net_vega)
    add('max_abs_net_theta', metrics['net_theta'], cfg.max_abs_net_theta)
    add('max_flat_7d_loss_abs', metrics['flat_7d_loss'], cfg.max_flat_7d_loss_abs)
    add('max_down_10_iv_down_10_loss_abs', metrics['down_10_iv_down_10_7d_loss'], cfg.max_down_10_iv_down_10_loss_abs)
    add('max_same_expiry_positions', metrics['max_same_expiry_count'], cfg.max_same_expiry_positions, 'gt')
    add('max_same_option_type_positions', metrics['max_same_option_type_count'], cfg.max_same_option_type_positions, 'gt')
    report = pd.DataFrame([{**metrics, 'breach_count': len(breaches), 'risk_status': 'BREACH' if breaches else 'OK'}])
    bdf = pd.DataFrame(breaches)
    report.to_csv(cfg.report_file, index=False)
    bdf.to_csv(cfg.breaches_file, index=False)
    print(f"Portfolio risk-limit check complete. status={report.iloc[-1]['risk_status']} breaches={len(breaches)}")
    return report, bdf


if __name__ == '__main__':
    check_portfolio_risk_limits()
