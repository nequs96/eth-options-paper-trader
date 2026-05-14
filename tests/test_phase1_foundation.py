import tempfile
from pathlib import Path

import pandas as pd

from data.data_quality import validate_option_chain
from storage.database import initialize_database, query
from storage.market_data_repository import save_option_chain_snapshot


def test_option_chain_quality_and_database_write():
    with tempfile.TemporaryDirectory() as tmp:
        db = str(Path(tmp) / "test.db")
        initialize_database(db)
        data = pd.DataFrame([
            {
                "instrument_name": "ETH-TEST-2500-C",
                "option_type": "call",
                "strike": 2500,
                "days_to_expiry": 10,
                "underlying_price_usd": 2400,
                "mark_price_usd": 100,
                "bid_price_usd": 98,
                "ask_price_usd": 102,
                "implied_volatility": 0.70,
            }
        ])
        report = validate_option_chain(data)
        assert report.quality_score == 1.0
        snapshot_id = save_option_chain_snapshot(data, "2026-01-01T00:00:00Z", db)
        assert snapshot_id > 0
        rows = query(db, "SELECT COUNT(*) AS n FROM option_quotes")
        assert rows[0]["n"] == 1
