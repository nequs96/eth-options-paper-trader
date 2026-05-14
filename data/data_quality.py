from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import json
import pandas as pd


@dataclass(frozen=True)
class DataQualityReport:
    dataset_name: str
    total_rows: int
    valid_rows: int
    invalid_rows: int
    quality_score: float
    missing_required_count: int
    invalid_price_count: int
    crossed_market_count: int
    wide_spread_count: int
    invalid_iv_count: int
    duplicate_instrument_count: int
    stale_or_missing_timestamp_count: int
    notes: str
    issues: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data['issues_json'] = json.dumps(self.issues, default=str)
        return data


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors='coerce') if col in df.columns else pd.Series([pd.NA] * len(df), index=df.index)


def _missing(df: pd.DataFrame, required: list[str]) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=bool)
    mask = pd.Series(False, index=df.index)
    for col in required:
        if col not in df.columns:
            return pd.Series(True, index=df.index)
        mask = mask | df[col].isna() | df[col].astype(str).str.strip().eq('')
    return mask


def validate_option_chain(option_chain: pd.DataFrame, max_spread_pct: float = 0.50) -> DataQualityReport:
    total = len(option_chain)
    if total == 0:
        return DataQualityReport('option_chain', 0, 0, 0, 0.0, 0, 0, 0, 0, 0, 0, 0, 'empty_dataset', [])
    required = ['instrument_name', 'option_type', 'strike', 'days_to_expiry', 'market_price_usd']
    missing = _missing(option_chain, required)
    price = _num(option_chain, 'market_price_usd')
    bid = _num(option_chain, 'bid_price_usd')
    ask = _num(option_chain, 'ask_price_usd')
    spread = _num(option_chain, 'bid_ask_spread_pct')
    iv = _num(option_chain, 'implied_volatility')
    invalid_price = price.isna() | (price <= 0)
    crossed = bid.notna() & ask.notna() & (bid > ask)
    wide = spread.notna() & (spread > max_spread_pct)
    bad_iv = iv.notna() & ((iv <= 0) | (iv > 5.0))
    duplicate_count = int(option_chain['instrument_name'].astype(str).duplicated().sum()) if 'instrument_name' in option_chain.columns else total
    invalid = missing | invalid_price | crossed | wide | bad_iv
    invalid_rows = int(invalid.sum())
    valid_rows = total - invalid_rows
    score = valid_rows / total if total else 0.0
    counts = {'missing_required': int(missing.sum()), 'invalid_price': int(invalid_price.sum()), 'crossed_market': int(crossed.sum()), 'wide_spread': int(wide.sum()), 'invalid_iv': int(bad_iv.sum()), 'duplicate_instrument': duplicate_count}
    issues = [{'reason': k, 'count': v} for k, v in counts.items() if v]
    return DataQualityReport('option_chain', total, valid_rows, invalid_rows, float(score), counts['missing_required'], counts['invalid_price'], counts['crossed_market'], counts['wide_spread'], counts['invalid_iv'], duplicate_count, 0, 'ok' if score >= 0.95 and duplicate_count == 0 else 'review_required', issues)


def validate_candidates(candidates: pd.DataFrame) -> DataQualityReport:
    if candidates.empty:
        return DataQualityReport('candidate_signals', 0, 0, 0, 0.0, 0, 0, 0, 0, 0, 0, 0, 'empty_dataset', [])
    report = validate_option_chain(candidates)
    return DataQualityReport('candidate_signals', report.total_rows, report.valid_rows, report.invalid_rows, report.quality_score, report.missing_required_count, report.invalid_price_count, report.crossed_market_count, report.wide_spread_count, report.invalid_iv_count, report.duplicate_instrument_count, 0, report.notes, report.issues)


def save_quality_report(report: DataQualityReport, output_file: str) -> pd.DataFrame:
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([{**report.to_dict(), 'timestamp': pd.Timestamp.utcnow().isoformat()}])
    path = Path(output_file)
    if path.exists() and path.stat().st_size > 0:
        try:
            row = pd.concat([pd.read_csv(path), row], ignore_index=True)
        except Exception:
            pass
    row.to_csv(path, index=False)
    return row
