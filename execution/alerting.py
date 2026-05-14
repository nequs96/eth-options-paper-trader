from __future__ import annotations
from pathlib import Path
import pandas as pd

def record_alert(message: str, severity: str='INFO', category: str='system', output_file: str='outputs/alerts_log.csv'):
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    row=pd.DataFrame([{'timestamp':pd.Timestamp.utcnow().isoformat(),'severity':severity,'category':category,'message':message}])
    p=Path(output_file)
    if p.exists() and p.stat().st_size>0:
        try: row=pd.concat([pd.read_csv(p),row], ignore_index=True)
        except Exception: pass
    row.to_csv(p,index=False); print(f'ALERT[{severity}] {category}: {message}')
    return row
if __name__=='__main__': record_alert('alerting module ready')
