from __future__ import annotations
from pathlib import Path
import sys
PROJECT_ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(PROJECT_ROOT))
from execution.scheduler_risk_guard import evaluate_scheduler_entry_guard
from execution.portfolio_derisker import build_derisking_recommendations
from storage.snapshot_store import record_snapshot
from backtesting.historical_replay_engine import run_historical_replay
from execution.realistic_order_simulator import simulate_realistic_orders
from execution.risk_budget_sizer import generate_risk_budget_sizing
from execution.smart_exit_executor import execute_smart_exits_paper
from execution.alerting import record_alert
from execution.regime_policy import build_regime_policy
from execution.exposure_balancer import build_exposure_balance_report
from execution.live_safety_layer import check_live_safety

def main():
    guard=evaluate_scheduler_entry_guard(); build_derisking_recommendations(); record_snapshot(); run_historical_replay(); simulate_realistic_orders(); generate_risk_budget_sizing(); execute_smart_exits_paper(auto_execute=False); build_regime_policy(); build_exposure_balance_report(); check_live_safety(); record_alert('Q8-Q20 suite completed','INFO','suite'); print('Q8_Q20_SUITE_COMPLETE')
if __name__=='__main__': main()
