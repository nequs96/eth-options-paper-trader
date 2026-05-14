from __future__ import annotations
from pathlib import Path
import sys, time, traceback
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from execution.q32_loss_control import evaluate_loss_controls
from execution.q32_call_throttle import evaluate_call_throttle
from execution.q32_profit_lock import execute_profit_lock_paper
from execution.q32_forced_derisk import execute_forced_derisk_paper
from execution.q32_candidate_hard_filter import apply_q32_candidate_filter
try:
    from scripts.run_guarded_overnight import run_guarded_cycle
except Exception:
    run_guarded_cycle = None


def run_q32_loop(sleep_seconds: int = 900, max_cycles: int | None = None, auto_execute_derisk: bool = False):
    cycle = 0
    while True:
        cycle += 1
        print(f'\n========== Q32 GUARDED PAPER LOOP cycle={cycle} ==========')
        try:
            loss = evaluate_loss_controls()
            evaluate_call_throttle()
            # Refresh/manage positions if guarded runner exists; block entries if loss control halted.
            open_allowed = loss.get('action') == 'ALLOW'
            if run_guarded_cycle is not None:
                run_guarded_cycle(open_when_allowed=open_allowed)
            else:
                print('WARNING: run_guarded_cycle unavailable; running Q32 controls only.')
            apply_q32_candidate_filter()
            execute_profit_lock_paper(auto_execute=auto_execute_derisk)
            execute_forced_derisk_paper(auto_execute=auto_execute_derisk)
        except KeyboardInterrupt:
            print('Q32 loop stopped by user.')
            break
        except Exception:
            traceback.print_exc()
        if max_cycles is not None and cycle >= max_cycles:
            print(f'Reached max_cycles={max_cycles}')
            break
        print(f'Q32 loop sleeping {sleep_seconds}s...')
        time.sleep(sleep_seconds)

if __name__ == '__main__':
    run_q32_loop(auto_execute_derisk='--execute-derisk' in sys.argv)
