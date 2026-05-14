from __future__ import annotations
from pathlib import Path
import py_compile
import sys
import tempfile
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

FILES = [
    'models/black_scholes.py','models/greeks.py','models/volatility.py','models/implied_volatility.py','models/market_confidence.py','models/eth_forward_volatility.py','models/trend_regime_model.py','models/vol_surface_diagnostics.py','models/surface_fitting.py','models/portfolio_greeks.py','models/execution_cost_model.py','models/scenario_risk.py','data/market_data.py','data/options_data.py','data/data_quality.py','strategies/option_mispricing.py','backtesting/portfolio.py','backtesting/live_option_backtest_engine.py','execution/dynamic_risk_sizing.py','execution/dynamic_exit_config.py','execution/hybrid_exit_rules.py','execution/paper_trader.py','execution/paper_position_manager.py','execution/paper_account_reconciliation.py','execution/paper_equity_curve.py','execution/paper_risk_metrics.py','execution/paper_performance_report.py','execution/runtime_healthcheck.py','execution/professional_candidate_filter.py','execution/clean_reporting.py','execution/live_scheduler.py','execution/institutional_recorder.py','execution/surface_candidate_scorer.py','execution/order_simulator.py','execution/parameter_sweep.py','execution/portfolio_optimizer.py','execution/trade_attribution.py','storage/database.py','storage/market_data_repository.py','visualization/institutional_dashboard.py','scripts/run_institutional_research_suite.py','scripts/record_current_state.py'
]

def main() -> None:
    ok = True
    for rel in FILES:
        try:
            py_compile.compile(str(PROJECT_ROOT / rel), doraise=True)
            print(f'OK   {rel}')
        except Exception as error:
            ok = False
            print(f'FAIL {rel}: {error}')
    if not ok:
        raise SystemExit(1)
    from models.black_scholes import black_scholes_price
    assert black_scholes_price(2000, 2000, 30/365, 0.04, 0.75, 'call') > 0
    from models.greeks import all_greeks
    assert 'delta' in all_greeks(2000, 2000, 30/365, 0.04, 0.75, 'call')
    from data.data_quality import validate_option_chain
    sample = pd.DataFrame([{'instrument_name':'ETH-TEST-2500-C','option_type':'call','strike':2500,'days_to_expiry':10,'market_price_usd':100,'bid_price_usd':98,'ask_price_usd':102,'implied_volatility':0.7}])
    assert validate_option_chain(sample).quality_score == 1.0
    from storage.database import initialize_database, query
    with tempfile.TemporaryDirectory() as tmp:
        db = str(Path(tmp) / 'test.db')
        initialize_database(db)
        rows = query(db, "SELECT name FROM sqlite_master WHERE type='table'")
        assert rows
    from execution.live_scheduler import LiveSchedulerConfig
    required = ['max_option_chain_instruments','ticker_timeout_seconds','option_chain_progress_every','save_partial_option_chain','run_healthcheck','strict_healthcheck','fallback_historical_volatility','clean_terminal_report']
    missing = [field for field in required if field not in LiveSchedulerConfig.__dataclass_fields__]
    if missing:
        raise AssertionError(f'Missing scheduler fields: {missing}')
    print('MASTER_AUDIT_OK')

if __name__ == '__main__':
    main()
