from __future__ import annotations
from pathlib import Path
import py_compile, sys
PROJECT_ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(PROJECT_ROOT))
FILES=['execution/autonomous_config.py','execution/robust_io.py','execution/schema_validator.py','execution/stale_data_guard.py','execution/cycle_checkpoint.py','execution/safety_kernel.py','execution/autonomous_derisk_executor.py','execution/regime_failure_detector.py','execution/autonomous_pause_resume.py','execution/self_termination_guard.py','scripts/safe_mode.py','scripts/autonomous_paper_daemon.py','scripts/log_scanner.py']
for rel in FILES:
    py_compile.compile(str(PROJECT_ROOT/rel), doraise=True); print(f'OK   {rel}')
from execution.safety_kernel import SafetyKernel
assert SafetyKernel().allow('reporting').action=='reporting'
print('AUTONOMOUS_SAFETY_AUDIT_OK')
