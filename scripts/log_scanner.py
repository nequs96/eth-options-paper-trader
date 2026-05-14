from __future__ import annotations
from pathlib import Path
import pandas as pd
from execution.robust_io import atomic_write_csv
PATTERNS=['traceback','error','exception','fail','operationalerror','schema','risk breach']
def scan_logs(log_file='logs/guarded_overnight.log', output_file='outputs/log_scan_report.csv'):
    p=Path(log_file); rows=[]
    if p.exists():
        for i,line in enumerate(p.read_text(errors='ignore').splitlines(),1):
            low=line.lower()
            if any(x in low for x in PATTERNS): rows.append({'line_number':i,'text':line[:500]})
    out=pd.DataFrame(rows); atomic_write_csv(output_file,out); print(f'Log scan complete. matches={len(out)}'); return out
if __name__=='__main__': scan_logs()
