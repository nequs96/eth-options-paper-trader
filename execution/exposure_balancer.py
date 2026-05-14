from __future__ import annotations
from pathlib import Path
import pandas as pd
from execution.common_io import load_csv

def build_exposure_balance_report(positions_file='outputs/paper_open_positions.csv', output_file='outputs/exposure_balance_report.csv'):
    df=load_csv(positions_file); Path('outputs').mkdir(exist_ok=True)
    if not df.empty and 'status' in df.columns: df=df[df['status'].astype(str).str.lower().eq('open')]
    rows=[]
    if not df.empty:
        if 'option_type' in df.columns:
            for k,v in df['option_type'].astype(str).value_counts().items(): rows.append({'bucket':'option_type','name':k,'count':int(v),'limit':5,'breach':v>5})
        if 'expiry' in df.columns:
            for k,v in df['expiry'].astype(str).value_counts().items(): rows.append({'bucket':'expiry','name':k,'count':int(v),'limit':3,'breach':v>3})
    out=pd.DataFrame(rows); out.to_csv(output_file,index=False); print(f'Exposure balance report complete rows={len(out)}'); return out
if __name__=='__main__': build_exposure_balance_report()
