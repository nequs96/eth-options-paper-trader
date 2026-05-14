from __future__ import annotations
from pathlib import Path
import py_compile, sys, tempfile
import pandas as pd
PROJECT_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT_ROOT))
FILES=['execution/walk_forward_validation.py','execution/portfolio_risk_limits.py','execution/smart_exit_engine.py','visualization/enhanced_research_dashboard.py','scripts/run_q4_q7_suite.py','scripts/audit_q4_q7.py']
for rel in FILES:
    py_compile.compile(str(PROJECT_ROOT/rel), doraise=True); print(f'OK   {rel}')
from execution.walk_forward_validation import WalkForwardConfig, run_walk_forward_validation
from execution.portfolio_risk_limits import PortfolioRiskLimitConfig, check_portfolio_risk_limits
from execution.smart_exit_engine import SmartExitConfig, generate_smart_exit_recommendations
from visualization.enhanced_research_dashboard import generate_enhanced_research_dashboard
with tempfile.TemporaryDirectory() as tmp:
    t=Path(tmp)
    cand=pd.DataFrame([{'instrument_name':'GOOD','classification':'cheap','institutional_edge_score':0.55,'days_to_expiry':14,'bid_ask_spread_pct':0.03,'abs_moneyness':0.04,'price_diff_pct':-0.2},{'instrument_name':'BAD','classification':'cheap','institutional_edge_score':0.30,'days_to_expiry':3,'bid_ask_spread_pct':0.20,'abs_moneyness':0.30,'price_diff_pct':-0.1}])
    pos=pd.DataFrame([{'instrument_name':'GOOD','status':'open','option_type':'call','expiry':'2026-05-29','strike':2500,'days_to_expiry':1.5,'unrealized_pnl_pct':-0.35,'highest_profit_pct':0.1}])
    greeks=pd.DataFrame([{'net_delta':1.0,'net_gamma':0.01,'net_vega':10.0,'net_theta':-20.0}])
    scen=pd.DataFrame([{'spot_shock':0.0,'iv_shock':0.0,'days_forward':7,'portfolio_scenario_pnl':-100.0},{'spot_shock':-0.1,'iv_shock':-0.1,'days_forward':7,'portfolio_scenario_pnl':-200.0}])
    surf=pd.DataFrame([{'instrument_name':'GOOD','surface_relative_value_score':0.2}]); breaches=pd.DataFrame([{'limit_name':'dummy','value':1,'limit':0,'severity':'HARD'}])
    cf=t/'candidates.csv'; pf=t/'positions.csv'; gf=t/'greeks.csv'; sf=t/'scenarios.csv'; vf=t/'surface.csv'; bf=t/'breaches.csv'
    cand.to_csv(cf,index=False); pos.to_csv(pf,index=False); greeks.to_csv(gf,index=False); scen.to_csv(sf,index=False); surf.to_csv(vf,index=False); breaches.to_csv(bf,index=False)
    wf,_=run_walk_forward_validation(WalkForwardConfig(candidates_file=str(cf), results_file=str(t/'wf.csv'), summary_file=str(t/'wfs.csv'), stability_file=str(t/'stab.csv'))); assert not wf.empty
    rep,_=check_portfolio_risk_limits(PortfolioRiskLimitConfig(positions_file=str(pf), greeks_file=str(gf), scenario_file=str(sf), report_file=str(t/'limit.csv'), breaches_file=str(t/'risk.csv'))); assert not rep.empty
    exits=generate_smart_exit_recommendations(SmartExitConfig(positions_file=str(pf), candidates_file=str(cf), surface_file=str(vf), risk_breaches_file=str(bf), output_file=str(t/'exits.csv'))); assert not exits.empty and exits.iloc[0]['exit_action']!='HOLD'
    generate_enhanced_research_dashboard(str(t/'dash.html')); assert (t/'dash.html').exists()
print('Q4_Q7_AUDIT_OK')
