from __future__ import annotations
from dataclasses import asdict
import pandas as pd
from execution.q32_config import Q32ProfitRobustConfig
from execution.q32_io import load_csv, append_event
try:
    from execution.kill_switch import set_kill_switch
except Exception:
    def set_kill_switch(active: bool, reason: str = ''):
        pd.DataFrame([{'timestamp': pd.Timestamp.utcnow().isoformat(), 'kill_switch_active': active, 'reason': reason}]).to_csv('outputs/kill_switch_status.csv', index=False)


def evaluate_loss_controls(cfg: Q32ProfitRobustConfig | None = None) -> dict:
    cfg = cfg or Q32ProfitRobustConfig()
    perf, hist = load_csv('outputs/paper_performance_summary.csv'), load_csv('outputs/paper_trade_history.csv')
    reasons = []
    if not perf.empty:
        latest = perf.iloc[-1]
        actual = float(latest.get('actual_pnl_vs_start', 0) or 0)
        realized = float(latest.get('realized_pnl', 0) or 0)
        if actual <= -abs(cfg.max_total_drawdown_abs): reasons.append(f'total_drawdown_abs_exceeded:{actual:.2f}')
        if realized <= -abs(cfg.max_realized_loss_abs): reasons.append(f'realized_loss_abs_exceeded:{realized:.2f}')
    if not hist.empty and 'pnl_usd' in hist.columns:
        recent_sum = float(pd.to_numeric(hist['pnl_usd'], errors='coerce').fillna(0).tail(cfg.rolling_trade_window).sum())
        if recent_sum <= -abs(cfg.max_rolling_closed_trade_loss_abs): reasons.append(f'rolling_closed_trade_loss_exceeded:{recent_sum:.2f}')
    action = 'ALLOW' if not reasons else 'HALT_NEW_ENTRIES'
    if reasons: set_kill_switch(True, 'q32_loss_control:' + ';'.join(reasons))
    row = {'timestamp': pd.Timestamp.utcnow().isoformat(), 'action': action, 'reasons': ';'.join(reasons), **{f'cfg_{k}': v for k, v in asdict(cfg).items()}}
    append_event('outputs/q32_loss_control_report.csv', row)
    print(f'Q32 loss control: {action} {row["reasons"]}')
    return row

if __name__ == '__main__': evaluate_loss_controls()
