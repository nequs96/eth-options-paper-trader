from __future__ import annotations
from pathlib import Path
import pandas as pd


def set_kill_switch(active: bool, reason: str = '', file: str = 'outputs/kill_switch_status.csv') -> pd.DataFrame:
    Path(file).parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame([{
        'timestamp': pd.Timestamp.utcnow().isoformat(),
        'kill_switch_active': bool(active),
        'reason': str(reason),
    }])
    out.to_csv(file, index=False)
    print(f'Kill switch active={bool(active)} reason={reason}')
    return out


def is_kill_switch_active(file: str = 'outputs/kill_switch_status.csv') -> bool:
    p = Path(file)
    if not p.exists() or p.stat().st_size == 0:
        return False
    try:
        value = pd.read_csv(p).iloc[-1].get('kill_switch_active', False)
        return str(value).strip().lower() in {'true', '1', 'yes'}
    except Exception:
        return False


if __name__ == '__main__':
    set_kill_switch(False, 'initialized')
