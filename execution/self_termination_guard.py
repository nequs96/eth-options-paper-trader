from __future__ import annotations
from pathlib import Path
import pandas as pd
from execution.robust_io import read_csv_safe, atomic_write_json, append_csv_atomic
try:
    from execution.kill_switch import set_kill_switch
except Exception:
    def set_kill_switch(active: bool, reason: str=''): print('kill_switch', active, reason)

def evaluate_self_termination(max_error_rows:int=3, max_failed_status_rows:int=3, report_file='outputs/final_autonomous_report.json') -> bool:
    alerts=read_csv_safe('outputs/alerts_log.csv'); pause=read_csv_safe('outputs/autonomous_pause_state.csv'); safety=read_csv_safe('outputs/safety_kernel_decisions.csv')
    reasons=[]
    if not alerts.empty and 'severity' in alerts.columns and (alerts['severity'].astype(str).eq('ERROR').tail(max_error_rows).sum()>=max_error_rows): reasons.append('repeated_error_alerts')
    if not pause.empty and 'paused' in pause.columns and pause['paused'].astype(str).str.lower().eq('true').tail(max_failed_status_rows).sum()>=max_failed_status_rows: reasons.append('persistent_autonomous_pause')
    if not safety.empty and 'severity' in safety.columns and safety['severity'].astype(str).eq('HARD').tail(max_failed_status_rows).sum()>=max_failed_status_rows: reasons.append('persistent_safety_hard_blocks')
    terminate=bool(reasons)
    if terminate:
        set_kill_switch(True, ';'.join(reasons)); atomic_write_json(report_file, {'timestamp':pd.Timestamp.utcnow().isoformat(),'terminated':True,'reasons':reasons})
    append_csv_atomic('outputs/self_termination_guard_log.csv', {'timestamp':pd.Timestamp.utcnow().isoformat(),'terminate':terminate,'reasons':';'.join(reasons)})
    print(f'Self termination guard: terminate={terminate} reasons={reasons}')
    return terminate
if __name__=='__main__': evaluate_self_termination()
