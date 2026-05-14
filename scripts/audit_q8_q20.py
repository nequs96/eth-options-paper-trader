from __future__ import annotations
from pathlib import Path
import py_compile, sys, tempfile
import pandas as pd
PROJECT_ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(PROJECT_ROOT))
FILES=['execution/scheduler_risk_guard.py','scripts/run_guarded_overnight.py','execution/portfolio_derisker.py','storage/snapshot_store.py','backtesting/historical_replay_engine.py','models/slippage_model.py','execution/realistic_order_simulator.py','execution/risk_budget_sizer.py','execution/smart_exit_executor.py','execution/alerting.py','scripts/morning_report.py','models/regime_classifier.py','execution/regime_policy.py','execution/exposure_balancer.py','execution/live_safety_layer.py','execution/manual_approval_gate.py','execution/kill_switch.py','scripts/run_q8_q20_suite.py']
for rel in FILES:
    py_compile.compile(str(PROJECT_ROOT/rel), doraise=True); print(f'OK   {rel}')
from execution.live_safety_layer import check_live_safety
r=check_live_safety(); assert not r.empty and r.iloc[-1]['real_trading_allowed']==False
print('Q8_Q20_AUDIT_OK')
