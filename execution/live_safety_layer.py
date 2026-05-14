from __future__ import annotations
from dataclasses import dataclass, asdict
import pandas as pd

@dataclass(frozen=True)
class LiveSafetyConfig:
    paper_mode_lock: bool=True
    manual_approval_required: bool=True
    max_daily_loss: float=250.0
    max_orders_per_day: int=10
    kill_switch_file: str='outputs/kill_switch_status.csv'
    report_file: str='outputs/live_safety_report.csv'

def check_live_safety(config: LiveSafetyConfig|None=None) -> pd.DataFrame:
    cfg=config or LiveSafetyConfig()
    status='SAFE_PAPER_ONLY' if cfg.paper_mode_lock else 'REVIEW_REQUIRED'
    out=pd.DataFrame([{**asdict(cfg),'timestamp':pd.Timestamp.utcnow().isoformat(),'status':status,'real_trading_allowed':False}])
    out.to_csv(cfg.report_file,index=False); print(f'Live safety status={status}; real_trading_allowed=False'); return out
if __name__=='__main__': check_live_safety()
