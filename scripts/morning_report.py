from __future__ import annotations
from pathlib import Path
import sys, pandas as pd
PROJECT_ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(PROJECT_ROOT))
from execution.common_io import load_csv

def _last(path):
    df=load_csv(path); return {} if df.empty else df.tail(1).to_dict('records')[0]

def build_morning_report(output_md='outputs/morning_report.md', output_html='outputs/morning_report.html'):
    perf=_last('outputs/paper_performance_summary.csv'); risk=_last('outputs/portfolio_limit_report.csv'); opt=_last('outputs/optimizer_risk_gate_report.csv')
    exits=load_csv('outputs/smart_exit_recommendations.csv'); breaches=load_csv('outputs/risk_limit_breaches.csv')
    lines=['# Morning Report','',f"Generated: {pd.Timestamp.utcnow().isoformat()}",'','## Account',f"Estimated equity: {perf.get('estimated_equity','n/a')}",f"Total return: {perf.get('total_return','n/a')}",f"Open positions: {perf.get('open_positions','n/a')}",'','## Risk',f"Risk status: {risk.get('risk_status','n/a')}",f"Breach count: {risk.get('breach_count','n/a')}",f"Flat 7d loss: {risk.get('flat_7d_loss','n/a')}",'','## Optimizer',f"Status: {opt.get('status','n/a')}",f"Selected: {opt.get('selected','n/a')}",f"Reasons: {opt.get('portfolio_gate_reasons','')}",'',f"Smart exit review rows: {len(exits)}",f"Risk breach rows: {len(breaches)}"]
    md='\n'.join(lines); Path(output_md).write_text(md,encoding='utf-8')
    html='<html><body>'+''.join(f'<p>{line}</p>' for line in lines)+'</body></html>'; Path(output_html).write_text(html,encoding='utf-8')
    print(md); return md
if __name__=='__main__': build_morning_report()
