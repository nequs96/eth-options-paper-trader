from __future__ import annotations
from pathlib import Path
import pandas as pd

def queue_manual_approval(action: str, payload: str='', file: str='outputs/manual_approval_queue.csv'):
    Path(file).parent.mkdir(parents=True, exist_ok=True)
    row=pd.DataFrame([{'timestamp':pd.Timestamp.utcnow().isoformat(),'action':action,'payload':payload,'status':'PENDING'}])
    p=Path(file)
    if p.exists() and p.stat().st_size>0:
        try: row=pd.concat([pd.read_csv(p),row], ignore_index=True)
        except Exception: pass
    row.to_csv(p,index=False); return row
if __name__=='__main__': queue_manual_approval('test')
