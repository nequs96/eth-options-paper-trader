from __future__ import annotations
from pathlib import Path
import py_compile
import sys
import tempfile
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

FILES = [
    'execution/portfolio_optimizer.py',
    'scripts/audit_q1_optimizer.py',
]

for rel in FILES:
    py_compile.compile(str(PROJECT_ROOT / rel), doraise=True)
    print(f'OK   {rel}')

from execution.portfolio_optimizer import OptimizerRiskGateConfig, optimize_trade_list

with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    candidates = pd.DataFrame([
        {'instrument_name':'BAD-FRONT-WEEK','classification':'cheap','institutional_edge_score':0.55,'days_to_expiry':3.0,'bid_ask_spread_pct':0.03,'abs_moneyness':0.01,'open_interest':1000,'volume':200},
        {'instrument_name':'BAD-LIQUIDITY','classification':'cheap','institutional_edge_score':0.55,'days_to_expiry':14.0,'bid_ask_spread_pct':0.03,'abs_moneyness':0.01,'open_interest':2,'volume':2},
        {'instrument_name':'GOOD','classification':'cheap','institutional_edge_score':0.52,'days_to_expiry':14.0,'bid_ask_spread_pct':0.03,'abs_moneyness':0.04,'open_interest':500,'volume':100},
    ])
    scenarios = pd.DataFrame([{'spot_shock':0.0,'iv_shock':0.0,'days_forward':7,'portfolio_scenario_pnl':-100.0}])
    greeks = pd.DataFrame([{'net_delta':0.5,'net_gamma':0.0,'net_vega':10.0,'net_theta':-10.0}])
    candidates_file = tmp_path / 'candidates.csv'
    scenario_file = tmp_path / 'scenarios.csv'
    greeks_file = tmp_path / 'greeks.csv'
    output_file = tmp_path / 'optimized.csv'
    rejected_file = tmp_path / 'rejected.csv'
    report_file = tmp_path / 'report.csv'
    candidates.to_csv(candidates_file, index=False)
    scenarios.to_csv(scenario_file, index=False)
    greeks.to_csv(greeks_file, index=False)
    result = optimize_trade_list(config=OptimizerRiskGateConfig(
        candidates_file=str(candidates_file),
        scenario_file=str(scenario_file),
        greeks_file=str(greeks_file),
        output_file=str(output_file),
        rejected_file=str(rejected_file),
        report_file=str(report_file),
        max_positions=2,
    ))
    assert len(result) == 1
    assert result.iloc[0]['instrument_name'] == 'GOOD'
    rejected = pd.read_csv(rejected_file)
    assert len(rejected) == 2

print('Q1_OPTIMIZER_AUDIT_OK')
