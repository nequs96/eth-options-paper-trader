from __future__ import annotations
from pathlib import Path
import sys
PROJECT_ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(PROJECT_ROOT))
from execution.kill_switch import set_kill_switch
from execution.autonomous_derisk_executor import execute_autonomous_derisk
from execution.regime_failure_detector import detect_regime_failure
from execution.autonomous_pause_resume import update_pause_resume_state
from scripts.morning_report import build_morning_report
if __name__=='__main__':
    set_kill_switch(True,'safe_mode_enabled')
    detect_regime_failure(); update_pause_resume_state(); execute_autonomous_derisk(); build_morning_report(); print('SAFE_MODE_COMPLETE')
