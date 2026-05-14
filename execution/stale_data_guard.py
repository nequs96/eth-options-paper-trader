from __future__ import annotations
from pathlib import Path
import time
import pandas as pd
from execution.robust_io import atomic_write_csv

WATCH_FILES=['outputs/live_eth_option_chain.csv','outputs/optimized_trade_list.csv','outputs/portfolio_limit_report.csv','outputs/paper_open_positions.csv']

def check_stale_data(max_age_minutes: int=30, report_file='outputs/stale_data_report.csv') -> tuple[bool,pd.DataFrame]:
    now=time.time(); rows=[]; ok=True
    for file in WATCH_FILES:
        p=Path(file)
        if not p.exists() or p.stat().st_size==0:
            rows.append({'file':file,'exists':False,'age_minutes':None,'ok':False,'issue':'missing_or_empty'}); ok=False; continue
        age=(now-p.stat().st_mtime)/60
        good=age<=max_age_minutes
        if not good: ok=False
        rows.append({'file':file,'exists':True,'age_minutes':age,'ok':good,'issue':'ok' if good else 'stale'})
    df=pd.DataFrame(rows); atomic_write_csv(report_file,df); return ok,df

if __name__=='__main__':
    ok,_=check_stale_data(); print(f'STALE_DATA_OK={ok}')
