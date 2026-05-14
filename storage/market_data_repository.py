from __future__ import annotations
import json
from typing import Any
import pandas as pd
from data.data_quality import validate_option_chain
from storage.database import DEFAULT_DATABASE_FILE, execute_many, execute_one, initialize_database


def _sf(value: Any) -> float | None:
    try:
        result = float(value)
    except Exception:
        return None
    return result if pd.notna(result) else None


def _payload(row: pd.Series) -> str:
    return json.dumps(row.to_dict(), default=str)


def save_option_chain_snapshot(option_chain: pd.DataFrame, timestamp: str | None = None, database_file: str = DEFAULT_DATABASE_FILE) -> int:
    initialize_database(database_file)
    timestamp = timestamp or pd.Timestamp.utcnow().isoformat()
    report = validate_option_chain(option_chain)
    snapshot_id = execute_one(database_file, 'INSERT INTO option_chain_snapshots(timestamp, source, row_count, quality_score) VALUES (?, ?, ?, ?)', (timestamp, 'deribit', len(option_chain), report.quality_score))
    rows = []
    for _, row in option_chain.iterrows():
        rows.append((snapshot_id, row.get('instrument_name'), row.get('option_type'), row.get('expiry'), _sf(row.get('strike')), _sf(row.get('days_to_expiry')), _sf(row.get('underlying_price_usd', row.get('spot_price'))), _sf(row.get('market_price_usd')), _sf(row.get('bid_price_usd')), _sf(row.get('ask_price_usd')), _sf(row.get('mark_price_usd')), _sf(row.get('implied_volatility')), _sf(row.get('delta')), _sf(row.get('gamma')), _sf(row.get('theta')), _sf(row.get('vega')), _sf(row.get('open_interest')), _sf(row.get('volume')), _sf(row.get('bid_ask_spread_pct')), _sf(row.get('moneyness')), _sf(row.get('abs_moneyness'))))
    if rows:
        execute_many(database_file, 'INSERT INTO option_quotes(snapshot_id,instrument_name,option_type,expiry,strike,days_to_expiry,underlying_price_usd,market_price_usd,bid_price_usd,ask_price_usd,mark_price_usd,implied_volatility,delta,gamma,theta,vega,open_interest,volume,bid_ask_spread_pct,moneyness,abs_moneyness) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', rows)
    return snapshot_id


def save_candidate_signals(candidates: pd.DataFrame, rejected: pd.DataFrame | None = None, timestamp: str | None = None, database_file: str = DEFAULT_DATABASE_FILE) -> int:
    initialize_database(database_file)
    timestamp = timestamp or pd.Timestamp.utcnow().isoformat()
    frames = []
    if candidates is not None and not candidates.empty:
        frames.append(candidates.copy())
    if rejected is not None and not rejected.empty:
        frames.append(rejected.copy())
    if not frames:
        return 0
    data = pd.concat(frames, ignore_index=True)
    rows = []
    for _, row in data.iterrows():
        rows.append((timestamp, row.get('instrument_name'), row.get('option_type'), _sf(row.get('strike')), _sf(row.get('days_to_expiry')), _sf(row.get('market_price_usd')), _sf(row.get('model_price_usd', row.get('theoretical_price'))), _sf(row.get('price_diff_pct')), _sf(row.get('volatility_spread')), row.get('classification'), _sf(row.get('combined_score')), _sf(row.get('mci')), row.get('decision'), row.get('decision_reason'), row.get('raw_reject_reason', row.get('mci_reject_reason')), _payload(row)))
    execute_many(database_file, 'INSERT INTO candidate_signals(timestamp,instrument_name,option_type,strike,days_to_expiry,market_price_usd,model_price_usd,price_diff_pct,volatility_spread,classification,combined_score,mci,decision,decision_reason,reject_reason,payload_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', rows)
    return len(rows)


def save_paper_account_snapshot(snapshot: dict[str, Any], timestamp: str | None = None, database_file: str = DEFAULT_DATABASE_FILE) -> int:
    timestamp = timestamp or pd.Timestamp.utcnow().isoformat()
    return execute_one(database_file, 'INSERT INTO paper_account_snapshots(timestamp,cash,open_position_value,open_risk,unrealized_pnl,realized_pnl,estimated_equity,total_return,open_positions) VALUES (?,?,?,?,?,?,?,?,?)', (timestamp, _sf(snapshot.get('cash')), _sf(snapshot.get('open_position_value')), _sf(snapshot.get('open_risk')), _sf(snapshot.get('unrealized_pnl')), _sf(snapshot.get('realized_pnl', 0.0)), _sf(snapshot.get('estimated_equity')), _sf(snapshot.get('total_return')), int(snapshot.get('open_positions', 0))))


def save_position_snapshots(positions: pd.DataFrame, timestamp: str | None = None, database_file: str = DEFAULT_DATABASE_FILE) -> int:
    if positions is None or positions.empty:
        return 0
    timestamp = timestamp or pd.Timestamp.utcnow().isoformat()
    rows = []
    for _, row in positions.iterrows():
        rows.append((timestamp, row.get('instrument_name'), row.get('status'), row.get('option_type'), _sf(row.get('strike')), _sf(row.get('days_to_expiry')), _sf(row.get('entry_price_usd')), _sf(row.get('current_price_usd')), _sf(row.get('quantity')), _sf(row.get('capital_at_risk')), _sf(row.get('current_value_usd')), _sf(row.get('unrealized_pnl_usd')), _sf(row.get('unrealized_pnl_pct')), _payload(row)))
    execute_many(database_file, 'INSERT INTO paper_position_snapshots(timestamp,instrument_name,status,option_type,strike,days_to_expiry,entry_price_usd,current_price_usd,quantity,capital_at_risk,current_value_usd,unrealized_pnl_usd,unrealized_pnl_pct,payload_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)', rows)
    return len(rows)
