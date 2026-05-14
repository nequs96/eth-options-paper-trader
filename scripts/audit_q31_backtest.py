from __future__ import annotations
from pathlib import Path
import py_compile, tempfile
import pandas as pd
FILES=['backtesting/historical_options_schema.py','backtesting/historical_data_adapters.py','backtesting/full_strategy_replay.py','scripts/fetch_historical_options.py','scripts/run_6m_backtest.py','scripts/run_12m_backtest.py']
for f in FILES:
    py_compile.compile(f,doraise=True); print('OK  ',f)
from backtesting.historical_options_schema import normalize_options_dataframe
from backtesting.full_strategy_replay import ReplayConfig, run_full_strategy_replay
with tempfile.TemporaryDirectory() as tmp:
    p=Path(tmp)/'sample.csv'
    rows=[]
    for t in pd.date_range('2026-01-01', periods=5, freq='h', tz='UTC'):
        rows.append({'timestamp':t,'instrument_name':'ETH-29MAY26-2500-C','underlying_price_usd':2400,'bid_price_usd':40,'ask_price_usd':42,'mark_price_usd':41,'implied_volatility':0.5,'delta':0.3,'gamma':0.001,'vega':1.0,'theta':-2.0,'open_interest':1000,'volume':100})
    pd.DataFrame(rows).to_csv(p,index=False)
    df=normalize_options_dataframe(pd.read_csv(p)); assert not df.empty
    norm=Path(tmp)/'norm.csv'; df.to_csv(norm,index=False)
    summary=run_full_strategy_replay(ReplayConfig(input_file=str(norm), output_dir=str(Path(tmp)/'out'))); assert summary['timestamps']==5
print('Q31_BACKTEST_AUDIT_OK')
