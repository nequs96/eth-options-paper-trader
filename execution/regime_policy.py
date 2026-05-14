from __future__ import annotations
import pandas as pd

def build_regime_policy(regime: str='unknown', output_file: str='outputs/regime_policy_report.csv') -> pd.DataFrame:
    policy={'regime':regime,'allow_calls':True,'allow_puts':True,'min_score_adjustment':0.0,'max_theta_multiplier':1.0}
    if 'bear' in regime: policy.update({'allow_calls':False,'allow_puts':True,'min_score_adjustment':0.05})
    if 'bull' in regime: policy.update({'allow_calls':True,'allow_puts':False})
    out=pd.DataFrame([policy]); out.to_csv(output_file,index=False); print(f'Regime policy: {regime}'); return out
if __name__=='__main__': build_regime_policy()
