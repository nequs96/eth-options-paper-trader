from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import shutil, pandas as pd

@dataclass(frozen=True)
class SnapshotStoreConfig:
    source_files: tuple[str,...]=('outputs/live_eth_option_chain.csv','outputs/live_backtest_candidates_surface_scored.csv','outputs/optimized_trade_list.csv','outputs/portfolio_limit_report.csv','outputs/paper_open_positions.csv')
    snapshot_root: str='outputs/snapshots'
    manifest_file: str='outputs/snapshot_manifest.csv'

def record_snapshot(config: SnapshotStoreConfig|None=None) -> pd.DataFrame:
    cfg=config or SnapshotStoreConfig(); ts=pd.Timestamp.utcnow(); folder=Path(cfg.snapshot_root)/ts.strftime('%Y-%m-%d')/ts.strftime('%H%M%S')
    folder.mkdir(parents=True, exist_ok=True); rows=[]
    for f in cfg.source_files:
        p=Path(f)
        if p.exists() and p.stat().st_size>0:
            target=folder/p.name; shutil.copy2(p,target)
            rows.append({'timestamp':ts.isoformat(),'source_file':f,'snapshot_file':str(target),'size':target.stat().st_size})
    manifest=pd.DataFrame(rows); mf=Path(cfg.manifest_file); mf.parent.mkdir(parents=True,exist_ok=True)
    if mf.exists() and mf.stat().st_size>0:
        try: manifest=pd.concat([pd.read_csv(mf),manifest], ignore_index=True)
        except Exception: pass
    manifest.to_csv(mf,index=False); print(f'Snapshot recorded files={len(rows)} folder={folder}'); return manifest

if __name__=='__main__': record_snapshot()
