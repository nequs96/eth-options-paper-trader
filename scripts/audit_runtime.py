from __future__ import annotations

from pathlib import Path
import py_compile
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

FILES = [
    'data/options_data.py',
    'data/market_data.py',
    'backtesting/live_option_backtest_engine.py',
    'execution/runtime_healthcheck.py',
    'execution/live_scheduler.py',
    'execution/paper_trader.py',
    'execution/professional_candidate_filter.py',
    'execution/paper_position_manager.py',
    'execution/paper_account_reconciliation.py',
]

if __name__ == '__main__':
    ok = True
    for file_path in FILES:
        try:
            py_compile.compile(str(PROJECT_ROOT / file_path), doraise=True)
            print(f'OK   {file_path}')
        except Exception as error:
            ok = False
            print(f'FAIL {file_path}: {error}')
    if ok:
        from execution.live_scheduler import LiveSchedulerConfig
        required = ['max_option_chain_instruments', 'ticker_timeout_seconds', 'option_chain_progress_every', 'save_partial_option_chain', 'run_healthcheck', 'strict_healthcheck', 'fallback_historical_volatility']
        missing = [field for field in required if field not in LiveSchedulerConfig.__dataclass_fields__]
        if missing:
            ok = False
            print(f'FAIL LiveSchedulerConfig missing fields: {missing}')
        else:
            print('OK   LiveSchedulerConfig Phase 2A/v4 fields')
    raise SystemExit(0 if ok else 1)
