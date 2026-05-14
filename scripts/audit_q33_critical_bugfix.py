from __future__ import annotations
from pathlib import Path
import py_compile, sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
FILES = [
 'execution/common_io.py','execution/kill_switch.py','execution/schema_validator.py','execution/cycle_checkpoint.py',
 'models/portfolio_greeks.py','models/regime_classifier.py','strategies/risk_rules.py','execution/portfolio_risk_limits.py',
 'execution/paper_account_reconciliation.py','execution/q32_config.py','execution/q32_io.py','execution/q32_loss_control.py',
 'execution/q32_call_throttle.py','execution/q32_candidate_hard_filter.py','execution/q32_profit_lock.py','execution/q32_forced_derisk.py',
 'execution/smart_exit_executor.py','execution/autonomous_pause_resume.py','execution/autonomous_derisk_executor.py'
]
for rel in FILES:
    py_compile.compile(str(PROJECT_ROOT/rel), doraise=True)
    print('OK ', rel)
print('Q33_CRITICAL_BUGFIX_AUDIT_OK')
