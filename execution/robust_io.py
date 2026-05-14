from __future__ import annotations
from pathlib import Path
import json, os, tempfile
import pandas as pd

def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def atomic_write_text(path: str | Path, text: str) -> None:
    path=Path(path); ensure_parent(path)
    fd,tmp=tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=str(path.parent))
    with os.fdopen(fd,'w',encoding='utf-8') as f:
        f.write(text); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)

def atomic_write_json(path: str | Path, payload: dict) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, default=str))

def atomic_write_csv(path: str | Path, df: pd.DataFrame) -> None:
    path=Path(path); ensure_parent(path)
    fd,tmp=tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=str(path.parent))
    os.close(fd)
    df.to_csv(tmp,index=False)
    os.replace(tmp,path)

def read_csv_safe(path: str | Path) -> pd.DataFrame:
    p=Path(path)
    if not p.exists() or p.stat().st_size==0: return pd.DataFrame()
    try: return pd.read_csv(p)
    except Exception: return pd.DataFrame()

def append_csv_atomic(path: str | Path, row: dict) -> pd.DataFrame:
    old=read_csv_safe(path)
    new=pd.DataFrame([row])
    out=new if old.empty else pd.concat([old,new], ignore_index=True)
    atomic_write_csv(path,out)
    return out
