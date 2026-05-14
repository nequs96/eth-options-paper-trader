from __future__ import annotations
from pathlib import Path
import pandas as pd
from execution.common_io import load_csv, num

def run_historical_replay(manifest_file='outputs/snapshot_manifest.csv', output_file='outputs/historical_replay_results.csv', trades_file='outputs/historical_replay_trades.csv'):
    manifest=load_csv(manifest_file); rows=[]; trades=[]
    if manifest.empty:
        pd.DataFrame().to_csv(output_file,index=False); pd.DataFrame().to_csv(trades_file,index=False); print('Historical replay: no snapshots.'); return pd.DataFrame()
    for _,m in manifest.iterrows():
        sf=str(m.get('snapshot_file',''))
        if 'optimized_trade_list' not in sf: continue
        df=load_csv(sf)
        if df.empty: continue
        score=num(df,'optimizer_risk_adjusted_score', num(df,'institutional_edge_score',0.0))
        rows.append({'timestamp':m.get('timestamp'),'snapshot_file':sf,'selected':len(df),'mean_score':float(score.mean())})
        tmp=df.copy(); tmp['snapshot_timestamp']=m.get('timestamp'); trades.append(tmp)
    out=pd.DataFrame(rows); out.to_csv(output_file,index=False)
    all_trades=pd.concat(trades, ignore_index=True) if trades else pd.DataFrame(); all_trades.to_csv(trades_file,index=False)
    print(f'Historical replay complete. snapshots={len(out)} trades={len(all_trades)}'); return out
if __name__=='__main__': run_historical_replay()
