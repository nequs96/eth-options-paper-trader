from __future__ import annotations
from pathlib import Path
import pandas as pd
from execution.paper_trader import PaperTraderConfig, load_open_positions, load_cash, save_cash, save_open_positions, append_trade_history


def _safe_float(value, default: float = 0.0) -> float:
    try:
        result = float(value)
    except Exception:
        return default
    return result if pd.notna(result) else default


def _load_chain(path: str = 'outputs/live_eth_option_chain.csv') -> pd.DataFrame:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _price_lookup(chain: pd.DataFrame) -> dict:
    if chain.empty or 'instrument_name' not in chain.columns:
        return {}
    lookup = {}
    for _, row in chain.iterrows():
        price = _safe_float(row.get('market_price_usd'), 0.0) or _safe_float(row.get('mark_price_usd'), 0.0)
        if price > 0:
            lookup[str(row.get('instrument_name'))] = row.to_dict() | {'current_price_usd': price}
    return lookup


def manage_paper_positions(trader_config: PaperTraderConfig | None = None, **kwargs) -> pd.DataFrame:
    trader_config = trader_config or PaperTraderConfig()
    positions = load_open_positions(trader_config)
    if positions.empty:
        print('No open paper positions.')
        return pd.DataFrame()
    lookup = _price_lookup(_load_chain())
    open_rows, closed_rows = [], []
    cash = load_cash(trader_config)
    now = pd.Timestamp.utcnow().isoformat()
    for _, row in positions.iterrows():
        item = row.to_dict()
        update = lookup.get(str(item.get('instrument_name')))
        entry = _safe_float(item.get('entry_price_usd'), _safe_float(item.get('market_price_usd'), 0.0))
        qty = _safe_float(item.get('quantity'), 0.0)
        current = _safe_float(update.get('current_price_usd'), entry) if update else _safe_float(item.get('current_price_usd'), entry)
        if update:
            for col in ['underlying_price_usd','mark_price_usd','bid_price_usd','ask_price_usd','days_to_expiry','delta','gamma','theta','vega','implied_volatility']:
                if col in update and pd.notna(update[col]):
                    item[col] = update[col]
            item['last_update_status'] = 'updated'
        else:
            item['last_update_status'] = 'stale_no_chain_price'
        high = max(_safe_float(item.get('highest_price_usd'), entry), entry, current)
        pnl_pct = current / entry - 1.0 if entry > 0 else 0.0
        item.update({'entry_price_usd': entry, 'current_price_usd': current, 'highest_price_usd': high, 'current_value_usd': current * qty, 'unrealized_pnl_usd': (current - entry) * qty, 'unrealized_pnl_pct': pnl_pct, 'highest_profit_pct': high / entry - 1.0 if entry > 0 else 0.0, 'updated_at': now})
        dte = _safe_float(item.get('days_to_expiry'), 999.0)
        stop = _safe_float(item.get('dynamic_stop_loss_pct'), -0.25)
        hard_tp = _safe_float(item.get('dynamic_hard_take_profit_pct'), 0.80)
        if update and (dte <= _safe_float(item.get('dynamic_min_days_to_expiry_exit'), 1.5) or pnl_pct <= stop or pnl_pct >= hard_tp):
            item.update({'status': 'closed', 'closed_at': now, 'exit_price_usd': current, 'exit_value_usd': current * qty, 'pnl_usd': (current - entry) * qty, 'pnl_pct': pnl_pct, 'close_reason': 'risk_exit'})
            cash += current * qty
            closed_rows.append(item)
        else:
            item['status'] = 'open'
            open_rows.append(item)
    opened = pd.DataFrame(open_rows)
    closed = pd.DataFrame(closed_rows)
    save_open_positions(opened, trader_config)
    if not closed.empty:
        append_trade_history(closed, trader_config)
    save_cash(cash, trader_config)
    open_value = float(pd.to_numeric(opened.get('current_value_usd', pd.Series(dtype=float)), errors='coerce').fillna(0).sum()) if not opened.empty else 0.0
    unrealized = float(pd.to_numeric(opened.get('unrealized_pnl_usd', pd.Series(dtype=float)), errors='coerce').fillna(0).sum()) if not opened.empty else 0.0
    realized = float(pd.to_numeric(closed.get('pnl_usd', pd.Series(dtype=float)), errors='coerce').fillna(0).sum()) if not closed.empty else 0.0
    print(f'Position management done. Open={len(opened)}, Closed={len(closed)}, Cash=${cash:,.2f}, Open value=${open_value:,.2f}, Unrealized PnL=${unrealized:,.2f}, Realized this cycle=${realized:,.2f}')
    return opened


if __name__ == '__main__':
    manage_paper_positions()
