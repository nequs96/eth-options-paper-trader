from __future__ import annotations
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from execution.q32_loss_control import evaluate_loss_controls
from execution.q32_call_throttle import evaluate_call_throttle
from execution.q32_profit_lock import build_profit_lock_actions
from execution.q32_forced_derisk import build_forced_derisk_actions
from execution.q32_candidate_hard_filter import apply_q32_candidate_filter


def main():
    evaluate_loss_controls()
    evaluate_call_throttle()
    build_profit_lock_actions()
    build_forced_derisk_actions()
    apply_q32_candidate_filter()
    print('Q32_PROFITABILITY_SUITE_COMPLETE')

if __name__ == '__main__':
    main()
