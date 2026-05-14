from __future__ import annotations

from pathlib import Path
import py_compile
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

py_compile.compile(str(PROJECT_ROOT / 'backtesting/live_option_backtest_engine.py'), doraise=True)
from backtesting.live_option_backtest_engine import LiveOptionBacktestConfig
assert 'fallback_historical_volatility' in LiveOptionBacktestConfig.__dataclass_fields__
print('YAHOO429_PATCH_AUDIT_OK')
