from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import importlib
import sqlite3
import pandas as pd
from execution.paper_trader import PaperTraderConfig, load_csv_if_exists, load_cash


@dataclass
class HealthCheckItem:
    name: str
    ok: bool
    detail: str = ''


@dataclass
class HealthCheckResult:
    ok: bool
    items: list[HealthCheckItem]


def _check_import(module_name: str) -> HealthCheckItem:
    try:
        importlib.import_module(module_name)
        return HealthCheckItem(f'import:{module_name}', True, 'OK')
    except Exception as error:
        return HealthCheckItem(f'import:{module_name}', False, str(error))


def _check_output_folder(folder: str) -> HealthCheckItem:
    try:
        Path(folder).mkdir(parents=True, exist_ok=True)
        return HealthCheckItem('output_folder', True, folder)
    except Exception as error:
        return HealthCheckItem('output_folder', False, str(error))


def _check_cash(config: PaperTraderConfig) -> HealthCheckItem:
    try:
        cash = load_cash(config)
        return HealthCheckItem('paper_cash', cash >= -0.01, f'{cash:,.2f}')
    except Exception as error:
        return HealthCheckItem('paper_cash', False, str(error))


def _check_positions(config: PaperTraderConfig) -> HealthCheckItem:
    try:
        positions = load_csv_if_exists(config.positions_file)
        if positions.empty:
            return HealthCheckItem('positions_file', True, 'empty/no positions')
        if 'instrument_name' in positions.columns and 'status' in positions.columns:
            open_positions = positions[positions['status'].astype(str).str.lower().eq('open')]
            duplicates = int(open_positions['instrument_name'].astype(str).duplicated().sum())
            if duplicates > 0:
                return HealthCheckItem('positions_file', False, f'duplicate open instruments: {duplicates}')
        return HealthCheckItem('positions_file', True, f'rows={len(positions)}')
    except Exception as error:
        return HealthCheckItem('positions_file', False, str(error))


def _check_candidate_schema(config: PaperTraderConfig) -> HealthCheckItem:
    path = Path(config.candidates_file)
    if not path.exists() or path.stat().st_size == 0:
        return HealthCheckItem('candidate_schema', True, 'candidate file missing/empty before cycle')
    try:
        data = pd.read_csv(path, nrows=5)
    except pd.errors.EmptyDataError:
        return HealthCheckItem('candidate_schema', True, 'candidate file has no columns/rows')
    except Exception as error:
        return HealthCheckItem('candidate_schema', False, str(error))
    missing = sorted({'instrument_name', 'market_price_usd'} - set(data.columns))
    return HealthCheckItem('candidate_schema', not missing, f'missing columns: {missing}' if missing else f'columns={len(data.columns)}')


def _check_database(database_file: str = 'outputs/eth_options_research.db') -> HealthCheckItem:
    try:
        Path(database_file).parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(database_file)
        connection.execute('CREATE TABLE IF NOT EXISTS healthcheck_ping (timestamp TEXT)')
        connection.commit()
        connection.close()
        return HealthCheckItem('database', True, database_file)
    except Exception as error:
        return HealthCheckItem('database', False, str(error))


def print_healthcheck_report(result: HealthCheckResult) -> None:
    print('\n========== RUNTIME HEALTH CHECK =========')
    for item in result.items:
        print(f'{item.name:<48} {"OK" if item.ok else "FAIL":<5} {item.detail}')
    print(f'Result: {"PASS" if result.ok else "FAIL"}')
    print('=========================================')


def run_runtime_healthcheck(output_folder: str = 'outputs', paper_config: PaperTraderConfig | None = None, strict: bool = True) -> HealthCheckResult:
    paper_config = paper_config or PaperTraderConfig()
    items = [_check_output_folder(output_folder), _check_import('data.options_data'), _check_import('backtesting.live_option_backtest_engine'), _check_import('execution.paper_trader'), _check_import('execution.professional_candidate_filter'), _check_import('execution.paper_account_reconciliation'), _check_cash(paper_config), _check_positions(paper_config), _check_candidate_schema(paper_config), _check_database()]
    result = HealthCheckResult(all(item.ok for item in items), items)
    print_healthcheck_report(result)
    if strict and not result.ok:
        failed = '; '.join(f'{item.name}: {item.detail}' for item in items if not item.ok)
        raise RuntimeError('Runtime health check failed: ' + failed)
    return result
