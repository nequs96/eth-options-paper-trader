from __future__ import annotations
from pathlib import Path
import pandas as pd
try:
    from models.black_scholes import black_scholes_price
except Exception:
    black_scholes_price = None


def _intrinsic(spot, strike, option_type):
    return max(spot - strike, 0.0) if str(option_type).lower() == 'call' else max(strike - spot, 0.0)


def _price(spot, strike, dte, rate, vol, option_type):
    if black_scholes_price is None:
        return _intrinsic(spot, strike, option_type)
    try:
        return float(black_scholes_price(spot, strike, max(dte / 365.0, 1 / 365), rate, max(vol, 0.01), option_type))
    except Exception:
        return _intrinsic(spot, strike, option_type)


def run_position_scenarios(positions: pd.DataFrame, spot_shocks=None, iv_shocks=None, day_shocks=None, risk_free_rate: float = 0.04):
    spot_shocks = spot_shocks or [-0.20,-0.10,-0.05,0,0.05,0.10,0.20]
    iv_shocks = iv_shocks or [-0.10,0,0.10,0.20]
    day_shocks = day_shocks or [0,1,3,7]
    if positions is None or positions.empty:
        return pd.DataFrame(), pd.DataFrame()
    df = positions.copy()
    if 'status' in df.columns:
        df = df[df['status'].astype(str).str.lower().eq('open')]
    rows = []
    for _, row in df.iterrows():
        spot0 = float(row.get('underlying_price_usd', row.get('spot_price', 0)) or 0)
        strike = float(row.get('strike', 0) or 0)
        dte0 = float(row.get('days_to_expiry', 1) or 1)
        iv = float(row.get('implied_volatility', 0.75) or 0.75)
        qty = float(row.get('quantity', 0) or 0)
        current_value = float(row.get('current_value_usd', 0) or 0)
        option_type = row.get('option_type', 'call')
        for spot_shock in spot_shocks:
            for iv_shock in iv_shocks:
                for days_forward in day_shocks:
                    scenario_price = _price(spot0 * (1 + spot_shock), strike, max(dte0 - days_forward, 0.5), risk_free_rate, max(iv + iv_shock, 0.01), option_type)
                    scenario_value = scenario_price * qty
                    rows.append({'instrument_name': row.get('instrument_name'), 'spot_shock': spot_shock, 'iv_shock': iv_shock, 'days_forward': days_forward, 'scenario_price': scenario_price, 'scenario_value': scenario_value, 'scenario_pnl': scenario_value - current_value})
    pos = pd.DataFrame(rows)
    port = pos.groupby(['spot_shock','iv_shock','days_forward'], as_index=False)['scenario_pnl'].sum().rename(columns={'scenario_pnl':'portfolio_scenario_pnl'}) if not pos.empty else pd.DataFrame()
    return pos, port


def run_scenario_risk(positions_file: str = 'outputs/paper_open_positions.csv'):
    path = Path(positions_file)
    positions = pd.read_csv(path) if path.exists() and path.stat().st_size > 0 else pd.DataFrame()
    pos, port = run_position_scenarios(positions)
    Path('outputs').mkdir(exist_ok=True)
    pos.to_csv('outputs/scenario_position_pnl.csv', index=False)
    port.to_csv('outputs/scenario_portfolio_pnl.csv', index=False)
    summary = pd.DataFrame([{'worst_scenario_pnl': port['portfolio_scenario_pnl'].min() if not port.empty else 0.0, 'best_scenario_pnl': port['portfolio_scenario_pnl'].max() if not port.empty else 0.0}])
    summary.to_csv('outputs/scenario_risk_summary.csv', index=False)
    print(f'Scenario risk complete. position_rows={len(pos)} portfolio_rows={len(port)}')
    return pos, port


if __name__ == '__main__':
    run_scenario_risk()
