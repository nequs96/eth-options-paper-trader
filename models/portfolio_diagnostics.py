from __future__ import annotations
import pandas as pd


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors='coerce').fillna(0.0) if col in df.columns else pd.Series([0.0] * len(df), index=df.index)


def calculate_portfolio_diagnostics(open_positions: pd.DataFrame, initial_cash: float = 10000.0) -> dict:
    if open_positions is None or open_positions.empty:
        return {'open_positions':0,'open_risk':0.0,'current_value':0.0,'unrealized_pnl':0.0,'open_risk_pct':0.0,'net_delta':0.0,'net_gamma':0.0,'net_vega':0.0,'net_theta':0.0,'calls':0,'puts':0}
    df = open_positions.copy()
    if 'status' in df.columns:
        df = df[df['status'].astype(str).str.lower().eq('open')]
    qty = _num(df, 'quantity')
    return {'open_positions':len(df),'open_risk':float(_num(df,'capital_at_risk').sum()),'current_value':float(_num(df,'current_value_usd').sum()),'unrealized_pnl':float(_num(df,'unrealized_pnl_usd').sum()),'open_risk_pct':float(_num(df,'capital_at_risk').sum()/initial_cash) if initial_cash else 0.0,'net_delta':float((_num(df,'delta')*qty).sum()),'net_gamma':float((_num(df,'gamma')*qty).sum()),'net_vega':float((_num(df,'vega')*qty).sum()),'net_theta':float((_num(df,'theta')*qty).sum()),'calls':int(df['option_type'].astype(str).str.lower().eq('call').sum()) if 'option_type' in df.columns else 0,'puts':int(df['option_type'].astype(str).str.lower().eq('put').sum()) if 'option_type' in df.columns else 0}


def save_portfolio_diagnostics(open_positions: pd.DataFrame, output_file: str = 'outputs/portfolio_diagnostics.csv', initial_cash: float = 10000.0) -> pd.DataFrame:
    row = calculate_portfolio_diagnostics(open_positions, initial_cash)
    row['timestamp'] = pd.Timestamp.utcnow().isoformat()
    out = pd.DataFrame([row])
    try:
        out = pd.concat([pd.read_csv(output_file), out], ignore_index=True)
    except Exception:
        pass
    out.to_csv(output_file, index=False)
    return out
