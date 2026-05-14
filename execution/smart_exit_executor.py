from __future__ import annotations
from pathlib import Path
import pandas as pd
from execution.common_io import load_csv, sf


def execute_smart_exits_paper(positions_file: str = 'outputs/paper_open_positions.csv', rec_file: str = 'outputs/smart_exit_recommendations.csv', history_file: str = 'outputs/paper_trade_history.csv', actions_file: str = 'outputs/smart_exit_actions.csv', auto_execute: bool = False) -> pd.DataFrame:
    pos, rec = load_csv(positions_file), load_csv(rec_file); Path('outputs').mkdir(exist_ok=True)
    if pos.empty or rec.empty or 'instrument_name' not in rec.columns:
        pd.DataFrame().to_csv(actions_file,index=False); print('Smart exit executor: no actions.'); return pd.DataFrame()
    if 'exit_action' in rec.columns:
        rec = rec[rec['exit_action'].astype(str).ne('HOLD')]
    to_close = set(rec['instrument_name'].astype(str))
    actions=[]; keep=[]; closed=[]; now=pd.Timestamp.utcnow().isoformat()
    for _,r in pos.iterrows():
        if str(r.get('instrument_name')) in to_close and str(r.get('status','open')).lower()=='open':
            row=r.to_dict(); row['recommended_close_at']=now; row['smart_exit_action']='WOULD_CLOSE' if not auto_execute else 'CLOSED'; actions.append(row)
            if auto_execute:
                row['status']='closed'; row['closed_at']=now; row['close_reason']='smart_exit_executor'; closed.append(row)
            else: keep.append(r.to_dict())
        else: keep.append(r.to_dict())
    act=pd.DataFrame(actions); act.to_csv(actions_file,index=False)
    if auto_execute:
        pd.DataFrame(keep).to_csv(positions_file,index=False); old=load_csv(history_file); hist=pd.concat([old,pd.DataFrame(closed)], ignore_index=True) if not old.empty else pd.DataFrame(closed); hist.to_csv(history_file,index=False)
    print(f'Smart exit executor complete. actions={len(act)} auto_execute={auto_execute}')
    return act

if __name__=='__main__': execute_smart_exits_paper(auto_execute=False)
