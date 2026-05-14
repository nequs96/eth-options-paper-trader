from __future__ import annotations
import pandas as pd
from execution.robust_io import read_csv_safe, atomic_write_csv

def detect_regime_failure(output_file='outputs/regime_failure_report.csv') -> pd.DataFrame:
    perf=read_csv_safe('outputs/paper_performance_summary.csv'); hist=read_csv_safe('outputs/paper_trade_history.csv'); execq=read_csv_safe('outputs/execution_quality_report.csv')
    reasons=[]; status='HEALTHY'
    if not perf.empty:
        tr=float(perf.iloc[-1].get('total_return',0) or 0)
        if tr < -0.03: reasons.append('drawdown_gt_3pct'); status='DEGRADED'
        if tr < -0.05: reasons.append('drawdown_gt_5pct'); status='FAILED'
    if not hist.empty and 'pnl_pct' in hist.columns:
        recent=pd.to_numeric(hist['pnl_pct'], errors='coerce').dropna().tail(10)
        if len(recent)>=5 and (recent<0).mean()>=0.7: reasons.append('recent_loss_rate_high'); status=max(status,'DEGRADED') if status!='FAILED' else status
    if not execq.empty and 'execution_reject_reason' in execq.columns:
        rej=execq['execution_reject_reason'].astype(str).ne('').mean()
        if rej>0.5: reasons.append('execution_rejection_rate_high'); status='DEGRADED' if status!='FAILED' else status
    out=pd.DataFrame([{'timestamp':pd.Timestamp.utcnow().isoformat(),'regime_status':status,'reasons':';'.join(reasons)}])
    atomic_write_csv(output_file,out); print(f'Regime failure detector: {status} {reasons}'); return out
if __name__=='__main__': detect_regime_failure()
