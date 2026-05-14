from __future__ import annotations

from pathlib import Path
import py_compile
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

for relative in ['execution/clean_reporting.py', 'execution/live_scheduler.py', 'scripts/print_clean_summary.py']:
    py_compile.compile(str(PROJECT_ROOT / relative), doraise=True)
    print(f'OK   {relative}')

from execution.live_scheduler import LiveSchedulerConfig
assert 'clean_terminal_report' in LiveSchedulerConfig.__dataclass_fields__
print('CLEAN_SUMMARY_PATCH_AUDIT_OK')
