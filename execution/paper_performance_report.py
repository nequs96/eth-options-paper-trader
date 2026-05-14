from __future__ import annotations
from pathlib import Path
import pandas as pd
from execution.paper_trader import PaperTraderConfig, load_csv_if_exists, load_cash, open_only
from execution.paper_account_reconciliation import calculate_open_current_value, calculate_unrealized_pnl, calculate_realized_pnl


def summarize_candidates(config: PaperTraderConfig, output_file: str = 'outputs/paper_available_candidates_passed.csv'):
    candidates = load_csv_if_exists(config.candidates_file)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output_file, index=False)
    return len(candidates), candidates


def generate_paper_performance_report(config: PaperTraderConfig | None = None, output_file: str = 'outputs/paper_performance_summary.csv') -> pd.DataFrame:
    config = config or PaperTraderConfig()
    positions = open_only(load_csv_if_exists(config.positions_file))
    history = load_csv_if_exists(config.trade_history_file)
    cash = load_cash(config)
    value = calculate_open_current_value(positions)
    n_candidates, _ = summarize_candidates(config)
    row = {'timestamp': pd.Timestamp.utcnow().isoformat(), 'cash': cash, 'open_position_value': value, 'estimated_equity': cash + value, 'realized_pnl': calculate_realized_pnl(history), 'unrealized_pnl': calculate_unrealized_pnl(positions), 'actual_pnl_vs_start': cash + value - config.initial_cash, 'total_return': (cash + value) / config.initial_cash - 1.0 if config.initial_cash else 0.0, 'open_positions': len(positions), 'available_candidates_passed': n_candidates, 'sharpe_ratio': 0.0, 'sortino_ratio': 0.0, 'profit_factor': 0.0, 'profit_ratio': 0.0}
    result = pd.DataFrame([row])
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_file, index=False)
    return result


def print_performance_report(report: pd.DataFrame) -> None:
    row = report.iloc[-1]
    print('\n========== PAPER PERFORMANCE =========')
    print(f"Estimated equity: ${float(row.get('estimated_equity', 0.0)):,.2f}")
    print(f"Total return: {float(row.get('total_return', 0.0)):.2%}")
    print(f"Available candidates passed: {int(row.get('available_candidates_passed', 0))}")
    print('======================================')
