from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from execution.robust_io import read_csv_safe, atomic_write_csv

REQUIRED_SCHEMAS = {
    'outputs/paper_open_positions.csv': ['instrument_name', 'status', 'current_price_usd'],
    'outputs/optimized_trade_list.csv': ['instrument_name', 'market_price_usd'],
    'outputs/portfolio_limit_report.csv': ['risk_status', 'breach_count'],
    'outputs/optimizer_risk_gate_report.csv': ['status', 'selected'],
}

@dataclass(frozen=True)
class SchemaValidationResult:
    ok: bool
    missing_files: list[str]
    missing_columns: dict[str, list[str]]


def validate_required_schemas(report_file: str = 'outputs/schema_validation_report.csv') -> SchemaValidationResult:
    rows = []
    missing_files: list[str] = []
    missing_cols: dict[str, list[str]] = {}

    for file, cols in REQUIRED_SCHEMAS.items():
        p = Path(file)
        if not p.exists() or p.stat().st_size == 0:
            missing_files.append(file)
            rows.append({'file': file, 'ok': False, 'issue': 'missing_or_empty', 'missing_columns': ''})
            continue
        df = read_csv_safe(file)
        miss = [c for c in cols if c not in df.columns]
        if miss:
            missing_cols[file] = miss
        rows.append({'file': file, 'ok': not miss, 'issue': 'missing_columns' if miss else 'ok', 'missing_columns': ';'.join(miss)})

    atomic_write_csv(report_file, pd.DataFrame(rows))
    return SchemaValidationResult(ok=(not missing_files and not missing_cols), missing_files=missing_files, missing_columns=missing_cols)


if __name__ == '__main__':
    r = validate_required_schemas()
    print(f'SCHEMA_VALIDATION_OK={r.ok}')
