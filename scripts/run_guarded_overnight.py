from __future__ import annotations
from pathlib import Path
import sys, time, traceback
PROJECT_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT_ROOT))
from backtesting.live_option_backtest_engine import LiveOptionBacktestConfig, run_live_option_paper_backtest
from execution.professional_candidate_filter import ProfessionalFilterConfig, filter_candidate_file
from execution.paper_trader import PaperTraderConfig, open_paper_trades
from execution.paper_position_manager import manage_paper_positions
from execution.paper_account_reconciliation import generate_reconciliation_report
from execution.paper_performance_report import generate_paper_performance_report, print_performance_report
from execution.paper_equity_curve import generate_equity_curve
from execution.paper_risk_metrics import calculate_risk_metrics, print_risk_metrics_report
from execution.clean_reporting import print_clean_cycle_report
from execution.scheduler_risk_guard import evaluate_scheduler_entry_guard


def run_guarded_cycle(max_option_chain_instruments=200, open_when_allowed=True):
    print('\n========== GUARDED PAPER CYCLE =========')
    pcfg=PaperTraderConfig(candidates_file='outputs/optimized_trade_list.csv', max_new_positions_per_cycle=2, target_positions=3, max_positions=8)
    manage_paper_positions(pcfg)
    generate_reconciliation_report(pcfg)
    run_live_option_paper_backtest(LiveOptionBacktestConfig(max_option_chain_instruments=max_option_chain_instruments, ticker_timeout_seconds=5, option_chain_progress_every=25))
    filter_candidate_file(ProfessionalFilterConfig())
    # optional surface scorer if available
    try:
        from models.vol_surface_diagnostics import run_surface_diagnostics
        from execution.surface_candidate_scorer import run_surface_candidate_scorer
        run_surface_diagnostics(); run_surface_candidate_scorer()
    except Exception as e: print(f'Surface scoring warning: {e}')
    guard=evaluate_scheduler_entry_guard()
    if guard.get('allow_new_entries') and open_when_allowed:
        open_paper_trades(pcfg)
    else:
        print('Guard blocked new entries; management/reporting only.')
    generate_reconciliation_report(pcfg)
    perf=generate_paper_performance_report(pcfg); generate_equity_curve(pcfg); risk=calculate_risk_metrics(config=pcfg)
    print_clean_cycle_report(pcfg); print_performance_report(perf); print_risk_metrics_report(risk)


def run_guarded_overnight(sleep_seconds=900, max_cycles=None, max_option_chain_instruments=200):
    cycle=0
    while True:
        cycle+=1
        try: run_guarded_cycle(max_option_chain_instruments=max_option_chain_instruments)
        except KeyboardInterrupt: print('\nGuarded scheduler stopped by user.'); break
        except Exception: print('ERROR in guarded cycle:'); traceback.print_exc()
        if max_cycles is not None and cycle>=max_cycles: print(f'Reached max_cycles={max_cycles}.'); break
        print(f'Sleeping {sleep_seconds} seconds...'); time.sleep(sleep_seconds)

if __name__=='__main__': run_guarded_overnight()
