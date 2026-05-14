import tempfile
from pathlib import Path
import pandas as pd
from models.market_confidence import clean_string, MarketConfidence
from execution.dynamic_risk_sizing import size_position, DynamicRiskConfig
from data.data_quality import validate_option_chain
from storage.database import initialize_database, query
from storage.market_data_repository import save_option_chain_snapshot

def test_nan_reject_reason_is_cleaned():
    assert clean_string(float('nan')) == ""
    assert clean_string("nan") == ""

def test_size_position_accepts_clean_candidate():
    mc = MarketConfidence(0.70,0.7,0.7,0.7,0.9,0.7,1.0,0.1,0.1,0.1,"")
    decision = size_position(10000,10000,50,mc,0.0,0.0,DynamicRiskConfig())
    assert decision.allowed
    assert decision.quantity > 0

def test_option_chain_quality_and_database_write():
    with tempfile.TemporaryDirectory() as tmp:
        db = str(Path(tmp) / "test.db")
        initialize_database(db)
        data = pd.DataFrame([{"instrument_name":"ETH-TEST-2500-C","option_type":"call","strike":2500,"days_to_expiry":10,"underlying_price_usd":2400,"mark_price_usd":100,"bid_price_usd":98,"ask_price_usd":102,"implied_volatility":0.70}])
        report = validate_option_chain(data)
        assert report.quality_score == 1.0
        snapshot_id = save_option_chain_snapshot(data, "2026-01-01T00:00:00Z", db)
        assert snapshot_id > 0
        rows = query(db, "SELECT COUNT(*) AS n FROM option_quotes")
        assert rows[0]["n"] == 1
