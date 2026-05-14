from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from backtesting.historical_options_schema import normalize_options_dataframe

@dataclass(frozen=True)
class ReplayConfig:
    input_file: str = 'data/historical_options/normalized.csv'
    output_dir: str = 'outputs/backtests'
    initial_cash: float = 10000.0
    max_positions: int = 8
    min_score: float = 0.55
    max_spread: float = 0.08
    min_dte: float = 7.0
    max_abs_moneyness: float = 0.12
    max_hold_steps: int = 24
    take_profit_pct: float = 0.50
    stop_loss_pct: float = -0.25

def _score(df: pd.DataFrame) -> pd.Series:
    # Proxy score for historical files. Full live model can replace this once provider has all fields.
    spread = pd.to_numeric(df.get('bid_ask_spread_pct',0.1), errors='coerce').fillna(0.1).clip(0,0.5)
    abs_m = pd.to_numeric(df.get('abs_moneyness',0.5), errors='coerce').fillna(0.5).clip(0,1)
    oi = pd.to_numeric(df.get('open_interest',0), errors='coerce').fillna(0).clip(0,10000)/10000
    vol = pd.to_numeric(df.get('volume',0), errors='coerce').fillna(0).clip(0,1000)/1000
    return (0.35*(1-spread/0.5)+0.30*(1-abs_m)+0.20*oi+0.15*vol).clip(0,1)

def run_full_strategy_replay(cfg: ReplayConfig|None=None) -> dict:
    cfg=cfg or ReplayConfig(); Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    raw=pd.read_csv(cfg.input_file)
    df=normalize_options_dataframe(raw) 
    if df.empty:
        raise RuntimeError(
            "Historical input is empty after normalization. "
            f"Input file: {cfg.input_file}. "
            f"Raw rows: {len(raw)}. "
            f"Raw columns: {list(raw.columns)}. "
            "This usually means the file has no rows, missing option columns, "
            "or no raw provider historical data was downloaded before normalization."
    )

    df['timestamp']=pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
    df=df.dropna(subset=['timestamp','instrument_name','market_price_usd']).sort_values('timestamp')
    if df.empty:
        raise RuntimeError(
            "Historical input had rows after normalization, but all rows were dropped "
            "because timestamp, instrument_name, or market_price_usd was missing/invalid."
    )
    df['replay_score']=_score(df)
    cash=cfg.initial_cash; positions=[]; trades=[]; equity=[]
    timestamps=list(df['timestamp'].dropna().sort_values().unique())
    for ts in timestamps:
        snap=df[df['timestamp']==ts].copy()
        # mark positions
        remaining=[]; pos_value=0.0
        for p in positions:
            q=snap[snap['instrument_name']==p['instrument_name']]
            price=float(q.iloc[0].get('bid_price_usd') if not q.empty and pd.notna(q.iloc[0].get('bid_price_usd')) else p['entry_price'])
            pnl_pct=(price-p['entry_price'])/p['entry_price'] if p['entry_price'] else 0
            p['age']+=1; p['current_price']=price; p['pnl_pct']=pnl_pct
            close_reason=None
            if pnl_pct>=cfg.take_profit_pct: close_reason='take_profit'
            elif pnl_pct<=cfg.stop_loss_pct: close_reason='stop_loss'
            elif p['age']>=cfg.max_hold_steps: close_reason='max_hold'
            if close_reason:
                cash += p['qty']*price
                p.update({'exit_time':ts,'exit_price':price,'pnl_usd':p['qty']*(price-p['entry_price']),'close_reason':close_reason})
                trades.append(p.copy())
            else:
                pos_value += p['qty']*price; remaining.append(p)
        positions=remaining
        # entries
        if len(positions)<cfg.max_positions:
            cand=snap[(snap['replay_score']>=cfg.min_score)&(snap['bid_ask_spread_pct'].fillna(999)<=cfg.max_spread)&(snap['abs_moneyness'].fillna(999)<=cfg.max_abs_moneyness)].sort_values('replay_score',ascending=False)
            open_names={p['instrument_name'] for p in positions}
            for _,r in cand.iterrows():
                if len(positions)>=cfg.max_positions: break
                inst=str(r['instrument_name'])
                if inst in open_names: continue
                entry=float(r.get('ask_price_usd') if pd.notna(r.get('ask_price_usd')) else r.get('market_price_usd'))
                if entry<=0 or cash<100: continue
                qty=100/entry; cash-=qty*entry
                positions.append({'instrument_name':inst,'entry_time':ts,'entry_price':entry,'qty':qty,'age':0,'option_type':r.get('option_type'),'strike':r.get('strike'),'expiry':r.get('expiry'),'entry_score':r.get('replay_score')})
                open_names.add(inst)
        equity.append({'timestamp':ts,'cash':cash,'position_value':pos_value,'equity':cash+pos_value,'open_positions':len(positions)})
    # close remaining at last mark
    for p in positions:
        trades.append({**p,'exit_time':timestamps[-1] if timestamps else None,'exit_price':p.get('current_price',p['entry_price']),'pnl_usd':p['qty']*(p.get('current_price',p['entry_price'])-p['entry_price']),'close_reason':'end_of_replay'})
    trades_df=pd.DataFrame(trades); eq_df=pd.DataFrame(equity)
    trades_df.to_csv(Path(cfg.output_dir)/'backtest_trades.csv',index=False)
    eq_df.to_csv(Path(cfg.output_dir)/'backtest_equity_curve.csv',index=False)
    summary={'initial_cash':cfg.initial_cash,'final_equity':float(eq_df.iloc[-1]['equity']) if not eq_df.empty else cfg.initial_cash,'total_return':float(eq_df.iloc[-1]['equity']/cfg.initial_cash-1) if not eq_df.empty else 0,'trades':len(trades_df),'timestamps':len(timestamps)}
    pd.DataFrame([summary]).to_csv(Path(cfg.output_dir)/'backtest_summary.csv',index=False)
    html='<html><body><h1>Historical Backtest Summary</h1>'+pd.DataFrame([summary]).to_html(index=False)+eq_df.tail(50).to_html(index=False)+'</body></html>'
    (Path(cfg.output_dir)/'backtest_summary.html').write_text(html,encoding='utf-8')
    print('BACKTEST_COMPLETE', summary)
    return summary
