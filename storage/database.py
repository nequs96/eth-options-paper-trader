from __future__ import annotations
from pathlib import Path
import sqlite3
from contextlib import contextmanager
from typing import Any, Iterable

DEFAULT_DATABASE_FILE = 'outputs/eth_options_research.db'

SCHEMA = [
    "CREATE TABLE IF NOT EXISTS system_events (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, event_type TEXT, severity TEXT, message TEXT, payload_json TEXT)",
    "CREATE TABLE IF NOT EXISTS option_chain_snapshots (snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, source TEXT, row_count INTEGER, quality_score REAL)",
    "CREATE TABLE IF NOT EXISTS option_quotes (id INTEGER PRIMARY KEY AUTOINCREMENT, snapshot_id INTEGER, instrument_name TEXT, option_type TEXT, expiry TEXT, strike REAL, days_to_expiry REAL, underlying_price_usd REAL, market_price_usd REAL, bid_price_usd REAL, ask_price_usd REAL, mark_price_usd REAL, implied_volatility REAL, delta REAL, gamma REAL, theta REAL, vega REAL, open_interest REAL, volume REAL, bid_ask_spread_pct REAL, moneyness REAL, abs_moneyness REAL)",
    "CREATE TABLE IF NOT EXISTS candidate_signals (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, instrument_name TEXT, option_type TEXT, strike REAL, days_to_expiry REAL, market_price_usd REAL, model_price_usd REAL, price_diff_pct REAL, volatility_spread REAL, classification TEXT, combined_score REAL, mci REAL, decision TEXT, decision_reason TEXT, reject_reason TEXT, payload_json TEXT)",
    "CREATE TABLE IF NOT EXISTS paper_account_snapshots (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, cash REAL, open_position_value REAL, open_risk REAL, unrealized_pnl REAL, realized_pnl REAL, estimated_equity REAL, total_return REAL, open_positions INTEGER)",
    "CREATE TABLE IF NOT EXISTS paper_position_snapshots (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, instrument_name TEXT, status TEXT, option_type TEXT, strike REAL, days_to_expiry REAL, entry_price_usd REAL, current_price_usd REAL, quantity REAL, capital_at_risk REAL, current_value_usd REAL, unrealized_pnl_usd REAL, unrealized_pnl_pct REAL, payload_json TEXT)",
]


@contextmanager
def connect(database_file: str = DEFAULT_DATABASE_FILE):
    Path(database_file).parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_file)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def initialize_database(database_file: str = DEFAULT_DATABASE_FILE) -> None:
    with connect(database_file) as connection:
        for statement in SCHEMA:
            connection.execute(statement)


def execute_one(database_file: str, sql: str, params: Iterable[Any] = ()) -> int:
    initialize_database(database_file)
    with connect(database_file) as connection:
        cursor = connection.execute(sql, tuple(params))
        return int(cursor.lastrowid)


def execute_many(database_file: str, sql: str, rows: Iterable[Iterable[Any]]) -> None:
    initialize_database(database_file)
    with connect(database_file) as connection:
        connection.executemany(sql, rows)


def query(database_file: str, sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
    initialize_database(database_file)
    with connect(database_file) as connection:
        return [dict(row) for row in connection.execute(sql, tuple(params)).fetchall()]
