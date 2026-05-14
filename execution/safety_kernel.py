from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from execution.autonomous_config import AutonomousSafetyConfig
from execution.robust_io import read_csv_safe, append_csv_atomic
from execution.schema_validator import validate_required_schemas
from execution.stale_data_guard import check_stale_data
try:
    from execution.kill_switch import is_kill_switch_active, set_kill_switch
except Exception:
    def is_kill_switch_active(*a,**k): return False
    def set_kill_switch(active: bool, reason: str='', file: str='outputs/kill_switch_status.csv'):
        pd.DataFrame([{'kill_switch_active':active,'reason':reason}]).to_csv(file,index=False)

@dataclass(frozen=True)
class SafetyDecision:
    allow: bool
    action: str
    reasons: list[str]
    severity: str

class SafetyKernel:
    def __init__(self, cfg: AutonomousSafetyConfig|None=None):
        self.cfg=cfg or AutonomousSafetyConfig()
    def _risk_reasons(self)->list[str]:
        r=[]; df=read_csv_safe('outputs/portfolio_limit_report.csv')
        if not df.empty:
            row=df.iloc[-1]
            if str(row.get('risk_status','')).upper()=='BREACH': r.append('portfolio_risk_breach')
            try:
                if abs(float(row.get('net_theta',0)))>self.cfg.max_abs_net_theta: r.append('theta_budget_exceeded')
                if abs(float(row.get('flat_7d_loss',0)))>self.cfg.max_flat_7d_loss_abs: r.append('flat_7d_loss_exceeded')
            except Exception: pass
        return r
    def allow(self, action: str) -> SafetyDecision:
        reasons=[]
        if self.cfg.real_trading_allowed: reasons.append('real_trading_not_allowed_by_autonomous_pack')
        if is_kill_switch_active(): reasons.append('kill_switch_active')
        schema=validate_required_schemas()
        if not schema.ok and action in {'open_trade','auto_exit','auto_derisk'}: reasons.append('schema_validation_failed')
        stale_ok,_=check_stale_data(self.cfg.stale_file_minutes)
        if not stale_ok and action in {'open_trade','auto_derisk'}: reasons.append('stale_data')
        reasons += self._risk_reasons() if action=='open_trade' else []
        severity='HARD' if reasons else 'OK'
        decision=SafetyDecision(allow=not reasons, action=action, reasons=reasons, severity=severity)
        append_csv_atomic('outputs/safety_kernel_decisions.csv', {'timestamp':pd.Timestamp.utcnow().isoformat(),'action':action,'allow':decision.allow,'reasons':';'.join(reasons),'severity':severity})
        return decision
    def fail_closed(self, reason: str):
        set_kill_switch(True, reason)
        append_csv_atomic('outputs/safety_kernel_decisions.csv', {'timestamp':pd.Timestamp.utcnow().isoformat(),'action':'fail_closed','allow':False,'reasons':reason,'severity':'HARD'})

if __name__=='__main__':
    d=SafetyKernel().allow('open_trade'); print(d)
