from __future__ import annotations
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from models.vol_surface_diagnostics import run_surface_diagnostics
from models.surface_fitting import run_surface_fit
from execution.surface_candidate_scorer import run_surface_candidate_scorer
from models.scenario_risk import run_scenario_risk
from models.portfolio_greeks import run_portfolio_greeks
from execution.parameter_sweep import run_parameter_sweep
from execution.trade_attribution import run_trade_attribution
from execution.portfolio_optimizer import optimize_trade_list
from execution.order_simulator import simulate_orders
from visualization.institutional_dashboard import generate_dashboard
try:
    from execution.institutional_recorder import record_full_cycle_outputs
except Exception:
    record_full_cycle_outputs = None

if __name__ == '__main__':
    run_surface_diagnostics()
    run_surface_fit()
    run_surface_candidate_scorer()
    run_scenario_risk()
    run_portfolio_greeks()
    run_parameter_sweep()
    run_trade_attribution()
    optimize_trade_list()
    simulate_orders()
    if record_full_cycle_outputs is not None:
        record_full_cycle_outputs()
    generate_dashboard()
    print('INSTITUTIONAL_RESEARCH_SUITE_COMPLETE')
