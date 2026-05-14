from __future__ import annotations
from pathlib import Path
import pandas as pd
from execution.paper_trader import load_csv_if_exists


def calculate_risk_metrics(equity_curve_file: str = 'outputs/paper_equity_curve.csv', output_file: str = 'outputs/paper_risk_metrics.csv', periods_per_year=None, config=None) -> pd.DataFrame:
    df = load_csv_if_exists(equity_curve_file)
    if df.empty or 'equity' not in df.columns or len(df) < 3:
        result = pd.DataFrame([{'timestamp': pd.Timestamp.utcnow().isoformat(), 'status': 'not_enough_data', 'sharpe_ratio': 0.0, 'profit_factor': 0.0}])
    else:
        equity = pd.to_numeric(df['equity'], errors='coerce').dropna()
        returns = equity.pct_change().dropna()
        sharpe = float(returns.mean() / returns.std(ddof=1) * (365 ** 0.5)) if returns.std(ddof=1) else 0.0
        result = pd.DataFrame([{'timestamp': pd.Timestamp.utcnow().isoformat(), 'status': 'ok', 'sharpe_ratio': sharpe, 'profit_factor': 0.0}])
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_file, index=False)
    return result


def print_risk_metrics_report(metrics: pd.DataFrame) -> None:
    print('\n========== PAPER RISK METRICS =========')
    if metrics.empty or metrics.iloc[-1].get('status') not in {'ok', 'not_enough_data'}:
        print('Not enough equity data yet.')
    else:
        print(f"Sharpe ratio: {float(metrics.iloc[-1].get('sharpe_ratio', 0.0)):.4f}")
        print(f"Profit factor: {float(metrics.iloc[-1].get('profit_factor', 0.0)):.4f}")
    print('=======================================')
