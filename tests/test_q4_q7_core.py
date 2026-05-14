from pathlib import Path
import tempfile
import pandas as pd
from execution.walk_forward_validation import WalkForwardConfig, run_walk_forward_validation
from execution.portfolio_risk_limits import PortfolioRiskLimitConfig, check_portfolio_risk_limits
from execution.smart_exit_engine import SmartExitConfig, generate_smart_exit_recommendations

def test_q4_q7_core_outputs():
    with tempfile.TemporaryDirectory() as tmp:
        t=Path(tmp)
        cand=pd.DataFrame([{'instrument_name':'GOOD','classification':'cheap','institutional_edge_score':0.55,'days_to_expiry':14,'bid_ask_spread_pct':0.03,'abs_moneyness':0.04,'price_diff_pct':-0.2}])
        pos=pd.DataFrame([{'instrument_name':'GOOD','status':'open','option_type':'call','expiry':'2026-05-29','strike':2500,'days_to_expiry':1.5,'unrealized_pnl_pct':-0.35,'highest_profit_pct':0.1}])
        greeks=pd.DataFrame([{'net_delta':1.0,'net_gamma':0.01,'net_vega':10.0,'net_theta':-20.0}])
        scen=pd.DataFrame([{'spot_shock':0.0,'iv_shock':0.0,'days_forward':7,'portfolio_scenario_pnl':-100.0}])
        cf=t/'candidates.csv'; pf=t/'positions.csv'; gf=t/'greeks.csv'; sf=t/'scenarios.csv'
        cand.to_csv(cf,index=False); pos.to_csv(pf,index=False); greeks.to_csv(gf,index=False); scen.to_csv(sf,index=False)
        wf,_=run_walk_forward_validation(WalkForwardConfig(candidates_file=str(cf), results_file=str(t/'wf.csv'), summary_file=str(t/'wfs.csv'), stability_file=str(t/'stab.csv'))); assert len(wf)>0
        report,_=check_portfolio_risk_limits(PortfolioRiskLimitConfig(positions_file=str(pf), greeks_file=str(gf), scenario_file=str(sf), report_file=str(t/'limit.csv'), breaches_file=str(t/'risk.csv'))); assert not report.empty
        exits=generate_smart_exit_recommendations(SmartExitConfig(positions_file=str(pf), candidates_file=str(cf), output_file=str(t/'exit.csv'))); assert len(exits)==1
