from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd
from execution.common_io import load_csv, ensure_outputs
try:
    from execution.portfolio_risk_limits import check_portfolio_risk_limits, PortfolioRiskLimitConfig
except Exception:
    check_portfolio_risk_limits = None
    PortfolioRiskLimitConfig = None
try:
    from execution.portfolio_optimizer import optimize_trade_list, OptimizerRiskGateConfig
except Exception:
    optimize_trade_list = None
    OptimizerRiskGateConfig = None

@dataclass(frozen=True)
class SchedulerRiskGuardConfig:
    max_open_positions: int = 8
    positions_file: str = 'outputs/paper_open_positions.csv'
    optimized_file: str = 'outputs/optimized_trade_list.csv'
    report_file: str = 'outputs/scheduler_risk_guard_report.csv'
    enforce_optimizer_entries: bool = True
    block_on_risk_breach: bool = True

def open_positions_count(path: str) -> int:
    df=load_csv(path)
    if df.empty: return 0
    if 'status' in df.columns: df=df[df['status'].astype(str).str.lower().eq('open')]
    return int(len(df))

def evaluate_scheduler_entry_guard(cfg: SchedulerRiskGuardConfig|None=None) -> dict:
    cfg=cfg or SchedulerRiskGuardConfig(); ensure_outputs()
    risk_status='UNKNOWN'; breach_count=0
    if check_portfolio_risk_limits is not None:
        report,_=check_portfolio_risk_limits()
        if not report.empty:
            risk_status=str(report.iloc[-1].get('risk_status','UNKNOWN'))
            breach_count=int(report.iloc[-1].get('breach_count',0))
    selected=0
    if optimize_trade_list is not None:
        try: optimize_trade_list()
        except Exception as e: print(f'Optimizer warning: {e}')
    opt=load_csv(cfg.optimized_file); selected=0 if opt.empty else len(opt)
    open_count=open_positions_count(cfg.positions_file)
    reasons=[]
    if cfg.block_on_risk_breach and risk_status=='BREACH': reasons.append('portfolio_risk_breach')
    if open_count>=cfg.max_open_positions: reasons.append('max_open_positions_reached')
    if cfg.enforce_optimizer_entries and selected<=0: reasons.append('optimizer_selected_zero')
    allow=not reasons
    row={**asdict(cfg),'timestamp':pd.Timestamp.utcnow().isoformat(),'allow_new_entries':allow,'block_reasons':';'.join(reasons),'risk_status':risk_status,'risk_breach_count':breach_count,'open_positions':open_count,'optimizer_selected':selected}
    out=pd.DataFrame([row]); path=Path(cfg.report_file)
    if path.exists() and path.stat().st_size>0:
        try: out=pd.concat([pd.read_csv(path),out], ignore_index=True)
        except Exception: pass
    out.to_csv(path,index=False)
    print(f'Scheduler risk guard: allow_new_entries={allow} reasons={row["block_reasons"] or "none"}')
    return row

if __name__=='__main__': evaluate_scheduler_entry_guard()
