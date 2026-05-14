from __future__ import annotations
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from execution.q32_loss_control import evaluate_loss_controls
from execution.q32_call_throttle import evaluate_call_throttle
from execution.q32_profit_lock import execute_profit_lock_paper
from execution.q32_forced_derisk import execute_forced_derisk_paper


def main(auto_execute: bool = False):
    evaluate_loss_controls()
    evaluate_call_throttle()
    execute_profit_lock_paper(auto_execute=auto_execute)
    execute_forced_derisk_paper(auto_execute=auto_execute)
    print(f'Q32_MANAGEMENT_ONLY_COMPLETE auto_execute={auto_execute}')

if __name__ == '__main__':
    # default is recommendation-only; pass --execute to mutate paper positions
    main(auto_execute='--execute' in sys.argv)
