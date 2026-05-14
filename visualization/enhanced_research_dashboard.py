from __future__ import annotations
from pathlib import Path
import html
import pandas as pd

def _load(path: str) -> pd.DataFrame:
    p=Path(path)
    if not p.exists() or p.stat().st_size==0: return pd.DataFrame()
    try: return pd.read_csv(p)
    except Exception: return pd.DataFrame()

def _section(title: str, data: pd.DataFrame, n: int = 25) -> str:
    body='<p>No data available.</p>' if data.empty else data.tail(n).to_html(index=False, escape=True)
    return '<section><h2>'+html.escape(title)+'</h2>'+body+'</section>'

def _cards(metrics: dict) -> str:
    return '<div class="cards">'+''.join('<div class="card"><div class="label">'+html.escape(str(k))+'</div><div class="value">'+html.escape(str(v))+'</div></div>' for k,v in metrics.items())+'</div>'

def generate_enhanced_research_dashboard(output_file: str = 'outputs/enhanced_research_dashboard.html') -> str:
    limit=_load('outputs/portfolio_limit_report.csv'); opt=_load('outputs/optimizer_risk_gate_report.csv'); greeks=_load('outputs/portfolio_greeks.csv')
    metrics={}
    if not limit.empty:
        r=limit.iloc[-1]; metrics.update({'risk_status':r.get('risk_status','unknown'),'breach_count':r.get('breach_count',0),'flat_7d_loss':r.get('flat_7d_loss',0)})
    if not opt.empty:
        r=opt.iloc[-1]; metrics.update({'optimizer_status':r.get('status','unknown'),'optimizer_selected':r.get('selected',0)})
    if not greeks.empty:
        r=greeks.iloc[-1]; metrics.update({'net_delta':round(float(r.get('net_delta',0)),4),'net_vega':round(float(r.get('net_vega',0)),4),'net_theta':round(float(r.get('net_theta',0)),4)})
    css='<style>body{font-family:Arial,sans-serif;margin:24px;background:#fafafa;color:#222}table{border-collapse:collapse;font-size:12px;background:white;width:100%}td,th{border:1px solid #ddd;padding:5px}th{background:#f0f0f0}.cards{display:flex;flex-wrap:wrap;gap:12px;margin:20px 0}.card{background:white;border:1px solid #ddd;border-radius:8px;padding:12px 16px;min-width:160px}.label{font-size:12px;color:#666}.value{font-size:20px;font-weight:bold}</style>'
    sections=[_cards(metrics),_section('Optimizer Risk Gate Report',opt),_section('Optimizer Rejected Candidates',_load('outputs/optimizer_rejected_candidates.csv')),_section('Portfolio Limit Report',limit),_section('Risk Limit Breaches',_load('outputs/risk_limit_breaches.csv')),_section('Smart Exit Recommendations',_load('outputs/smart_exit_recommendations.csv')),_section('Walk Forward Summary',_load('outputs/walk_forward_summary.csv')),_section('Parameter Stability Report',_load('outputs/parameter_stability_report.csv')),_section('Scenario Portfolio PnL',_load('outputs/scenario_portfolio_pnl.csv')),_section('Portfolio Greeks',greeks)]
    doc='<html><head><title>Enhanced Institutional Research Dashboard</title>'+css+'</head><body><h1>Enhanced Institutional Research Dashboard</h1><p>Q4-Q7 validation, risk limits, exits, and testing layer.</p>'+''.join(sections)+'</body></html>'
    Path(output_file).parent.mkdir(parents=True, exist_ok=True); Path(output_file).write_text(doc,encoding='utf-8'); print(f'Saved enhanced dashboard to: {output_file}'); return output_file

if __name__=='__main__': generate_enhanced_research_dashboard()
