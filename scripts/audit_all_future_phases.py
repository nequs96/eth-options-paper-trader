from pathlib import Path
import py_compile, sys
PROJECT_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT_ROOT))
FILES=['models/vol_surface_diagnostics.py','models/surface_fitting.py','execution/surface_candidate_scorer.py','models/scenario_risk.py','models/portfolio_greeks.py','models/execution_cost_model.py','execution/order_simulator.py','execution/parameter_sweep.py','execution/trade_attribution.py','execution/portfolio_optimizer.py','visualization/institutional_dashboard.py','scripts/run_institutional_research_suite.py']
for f in FILES:
    py_compile.compile(str(PROJECT_ROOT/f), doraise=True); print(f'OK   {f}')
print('ALL_FUTURE_PHASES_AUDIT_OK')
