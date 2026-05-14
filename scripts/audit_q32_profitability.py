from __future__ import annotations
from pathlib import Path
import py_compile, sys, tempfile
import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
FILES = [
 'execution/q32_config.py','execution/q32_io.py','execution/q32_loss_control.py','execution/q32_call_throttle.py',
 'execution/q32_profit_lock.py','execution/q32_forced_derisk.py','execution/q32_candidate_hard_filter.py',
 'scripts/run_q32_profitability_suite.py','scripts/run_q32_management_only_cycle.py','scripts/run_q32_guarded_paper_loop.py'
]
for rel in FILES:
    py_compile.compile(str(PROJECT_ROOT/rel), doraise=True)
    print('OK  ', rel)
from execution.q32_config import Q32ProfitRobustConfig
assert Q32ProfitRobustConfig().max_open_positions == 4
print('Q32_PROFITABILITY_AUDIT_OK')
