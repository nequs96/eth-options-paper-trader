from __future__ import annotations
from pathlib import Path
import pandas as pd
from execution.paper_trader import PaperTraderConfig, load_cash, load_open_positions

SUMMARY_COLUMNS = ['instrument_name','option_type','strike','days_to_expiry','entry_price_usd','current_price_usd','quantity','capital_at_risk','current_value_usd','unrealized_pnl_usd','unrealized_pnl_pct','status']


def _safe_float(value, default: float = 0.0) -> float:
    try:
        result = float(value)
    except Exception:
        return default
    return result if pd.notna(result) else default


def _money(value: float) -> str:
    return f'${value:,.2f}'


def _pct(value: float) -> str:
    return f'{value:.2%}'


def _num(data: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(data[column], errors='coerce').fillna(default) if column in data.columns else pd.Series([default] * len(data), index=data.index)


def summarize_open_positions(config: PaperTraderConfig | None = None) -> pd.DataFrame:
    config = config or PaperTraderConfig()
    data = load_open_positions(config).copy()
    if data.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    for column in SUMMARY_COLUMNS:
        if column not in data.columns:
            data[column] = '' if column in {'instrument_name','option_type','status'} else 0.0
    data['current_value_usd'] = _num(data, 'current_value_usd')
    if float(data['current_value_usd'].sum()) <= 0:
        data['current_value_usd'] = _num(data, 'current_price_usd') * _num(data, 'quantity')
    if 'unrealized_pnl_usd' not in data.columns or _num(data, 'unrealized_pnl_usd').abs().sum() == 0:
        data['unrealized_pnl_usd'] = (_num(data, 'current_price_usd') - _num(data, 'entry_price_usd')) * _num(data, 'quantity')
    entry = _num(data, 'entry_price_usd')
    current = _num(data, 'current_price_usd')
    data['unrealized_pnl_pct'] = (current / entry.replace(0, pd.NA) - 1.0).fillna(_num(data, 'unrealized_pnl_pct'))
    return data[SUMMARY_COLUMNS].sort_values(['days_to_expiry','instrument_name'], ascending=[True, True]).reset_index(drop=True)


def calculate_clean_account_snapshot(config: PaperTraderConfig | None = None) -> dict:
    config = config or PaperTraderConfig()
    cash = load_cash(config)
    positions = summarize_open_positions(config)
    open_value = float(_num(positions, 'current_value_usd').sum()) if not positions.empty else 0.0
    open_risk = float(_num(positions, 'capital_at_risk').sum()) if not positions.empty else 0.0
    unrealized = float(_num(positions, 'unrealized_pnl_usd').sum()) if not positions.empty else 0.0
    equity = cash + open_value
    return {'cash': cash, 'open_position_value': open_value, 'open_risk': open_risk, 'unrealized_pnl': unrealized, 'estimated_equity': equity, 'total_return': equity / config.initial_cash - 1.0 if config.initial_cash else 0.0, 'open_positions': int(len(positions))}


def print_clean_account_summary(config: PaperTraderConfig | None = None) -> None:
    snapshot = calculate_clean_account_snapshot(config)
    print('\n========== CLEAN PAPER ACCOUNT SUMMARY ==========')
    print(f"Free cash:              {_money(snapshot['cash'])}")
    print(f"Open position value:    {_money(snapshot['open_position_value'])}")
    print(f"Open risk/cost basis:   {_money(snapshot['open_risk'])}")
    print(f"Unrealized PnL:         {_money(snapshot['unrealized_pnl'])}")
    print(f"Estimated equity:       {_money(snapshot['estimated_equity'])}")
    print(f"Total return:           {_pct(snapshot['total_return'])}")
    print(f"Open positions:         {snapshot['open_positions']}")
    print('=================================================')


def print_clean_open_positions_table(config: PaperTraderConfig | None = None, max_rows: int = 20) -> None:
    positions = summarize_open_positions(config)
    print('\n========== CLEAN OPEN POSITIONS =========')
    if positions.empty:
        print('No open paper positions.')
        print('=========================================')
        return
    display = positions.head(max_rows).copy()
    for col in ['entry_price_usd','current_price_usd','capital_at_risk','current_value_usd','unrealized_pnl_usd']:
        display[col] = display[col].apply(lambda x: _money(_safe_float(x)))
    display['unrealized_pnl_pct'] = display['unrealized_pnl_pct'].apply(lambda x: _pct(_safe_float(x)))
    display = display.rename(columns={'instrument_name':'Instrument','option_type':'Type','strike':'Strike','days_to_expiry':'DTE','entry_price_usd':'Entry','current_price_usd':'Current','quantity':'Qty','capital_at_risk':'Risk','current_value_usd':'Value','unrealized_pnl_usd':'PnL $','unrealized_pnl_pct':'PnL %','status':'Status'})
    print(display.to_string(index=False))
    print('=========================================')


def save_clean_open_positions_summary(config: PaperTraderConfig | None = None, output_file: str = 'outputs/paper_open_positions_summary.csv') -> pd.DataFrame:
    summary = summarize_open_positions(config)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_file, index=False)
    return summary


def print_clean_cycle_report(config: PaperTraderConfig | None = None) -> None:
    print_clean_account_summary(config)
    print_clean_open_positions_table(config)
    save_clean_open_positions_summary(config)
