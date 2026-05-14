from __future__ import annotations
import pandas as pd
from execution.robust_io import read_csv_safe, atomic_write_csv


def update_pause_resume_state(output_file: str = 'outputs/autonomous_pause_state.csv') -> pd.DataFrame:
    risk = read_csv_safe('outputs/portfolio_limit_report.csv'); regime = read_csv_safe('outputs/regime_failure_report.csv'); prev = read_csv_safe(output_file)
    paused=False; reasons=[]
    if not risk.empty and str(risk.iloc[-1].get('risk_status','')).upper() == 'BREACH': paused=True; reasons.append('risk_breach')
    if not regime.empty and str(regime.iloc[-1].get('regime_status','')) in {'DEGRADED','FAILED'}: paused=True; reasons.append('regime_not_healthy')
    clean=0 if paused else (int(prev.iloc[-1].get('clean_cycles',0))+1 if not prev.empty else 1)
    out=pd.DataFrame([{'timestamp':pd.Timestamp.utcnow().isoformat(),'paused':paused,'reasons':';'.join(reasons),'clean_cycles':clean}])
    atomic_write_csv(output_file,out); print(f'Autonomous pause state: paused={paused} reasons={";".join(reasons)}'); return out

if __name__=='__main__': update_pause_resume_state()
