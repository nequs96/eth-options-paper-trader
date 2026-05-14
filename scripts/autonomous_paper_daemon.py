from __future__ import annotations
from pathlib import Path
import sys, time, traceback
PROJECT_ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(PROJECT_ROOT))
from execution.cycle_checkpoint import write_checkpoint
from execution.safety_kernel import SafetyKernel
from execution.autonomous_derisk_executor import execute_autonomous_derisk
from execution.regime_failure_detector import detect_regime_failure
from execution.autonomous_pause_resume import update_pause_resume_state
from execution.self_termination_guard import evaluate_self_termination
from execution.alerting import record_alert
try:
    from scripts.run_guarded_overnight import run_guarded_cycle
except Exception:
    run_guarded_cycle=None

def run_autonomous_daemon(sleep_seconds=900, max_cycles=None):
    cycle=0
    while True:
        cycle+=1; write_checkpoint('cycle_start','running',{'cycle':cycle})
        try:
            if evaluate_self_termination(): break
            detect_regime_failure(); pause=update_pause_resume_state(); paused=bool(pause.iloc[-1].get('paused',False)) if not pause.empty else False
            if paused:
                write_checkpoint('paused_derisk','running'); execute_autonomous_derisk()
            else:
                d=SafetyKernel().allow('open_trade')
                if run_guarded_cycle is not None: run_guarded_cycle(open_when_allowed=d.allow)
                else: record_alert('guarded cycle unavailable','ERROR','daemon')
            write_checkpoint('cycle_complete','ok',{'cycle':cycle})
        except KeyboardInterrupt:
            write_checkpoint('stopped_by_user','ok'); print('Autonomous paper daemon stopped by user.'); break
        except Exception as e:
            record_alert(f'autonomous daemon exception: {e}','ERROR','daemon'); traceback.print_exc(); write_checkpoint('cycle_exception','error',{'error':str(e)})
        if max_cycles is not None and cycle>=max_cycles: print(f'Reached max_cycles={max_cycles}'); break
        print(f'Autonomous daemon sleeping {sleep_seconds}s...'); time.sleep(sleep_seconds)
if __name__=='__main__': run_autonomous_daemon()
