from __future__ import annotations
from pathlib import Path
from typing import Any
import pandas as pd
from data.data_quality import validate_option_chain, validate_candidates, save_quality_report
from execution.clean_reporting import calculate_clean_account_snapshot, summarize_open_positions
from execution.paper_trader import PaperTraderConfig
from models.portfolio_diagnostics import save_portfolio_diagnostics
from storage.database import DEFAULT_DATABASE_FILE, initialize_database, execute_one
from storage.market_data_repository import save_candidate_signals, save_option_chain_snapshot, save_paper_account_snapshot, save_position_snapshots

DEFAULT_FILES = {'option_chain':'outputs/live_eth_option_chain.csv','raw_candidates':'outputs/live_backtest_candidates.csv','raw_rejected':'outputs/live_backtest_candidates_raw_rejected.csv','filtered_candidates':'outputs/live_backtest_candidates_filtered.csv','filtered_rejected':'outputs/live_backtest_candidates_rejected.csv','positions':'outputs/paper_open_positions.csv'}


def load_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def record_event(event_type: str, message: str, severity: str = 'INFO', payload: str | None = None, database_file: str = DEFAULT_DATABASE_FILE) -> int:
    initialize_database(database_file)
    return execute_one(database_file, 'INSERT INTO system_events(timestamp,event_type,severity,message,payload_json) VALUES (?,?,?,?,?)', (pd.Timestamp.utcnow().isoformat(), event_type, severity, message, payload))


def record_full_cycle_outputs(files: dict[str, str] | None = None, paper_config: PaperTraderConfig | None = None, database_file: str = DEFAULT_DATABASE_FILE) -> dict[str, Any]:
    files = files or DEFAULT_FILES
    paper_config = paper_config or PaperTraderConfig()
    initialize_database(database_file)
    timestamp = pd.Timestamp.utcnow().isoformat()
    chain = load_csv(files['option_chain'])
    raw = load_csv(files['raw_candidates'])
    raw_rej = load_csv(files['raw_rejected'])
    filt = load_csv(files['filtered_candidates'])
    filt_rej = load_csv(files['filtered_rejected'])
    option_report = validate_option_chain(chain)
    candidate_report = validate_candidates(raw)
    save_quality_report(option_report, 'outputs/data_quality_option_chain.csv')
    save_quality_report(candidate_report, 'outputs/data_quality_candidates.csv')
    snapshot_id = save_option_chain_snapshot(chain, timestamp, database_file) if not chain.empty else 0
    candidate_rows = save_candidate_signals(raw, raw_rej, timestamp, database_file)
    filtered_rows = save_candidate_signals(filt, filt_rej, timestamp, database_file)
    account = calculate_clean_account_snapshot(paper_config)
    account_id = save_paper_account_snapshot(account, timestamp, database_file)
    positions = summarize_open_positions(paper_config)
    position_rows = save_position_snapshots(positions, timestamp, database_file)
    save_portfolio_diagnostics(positions, initial_cash=paper_config.initial_cash)
    summary = {'timestamp':timestamp,'option_snapshot_id':snapshot_id,'candidate_rows_recorded':candidate_rows,'filtered_rows_recorded':filtered_rows,'account_snapshot_id':account_id,'position_rows_recorded':position_rows,'option_chain_rows':len(chain),'raw_candidates':len(raw),'filtered_candidates':len(filt),'open_positions':int(account.get('open_positions',0)),'estimated_equity':float(account.get('estimated_equity',0.0)),'total_return':float(account.get('total_return',0.0)),'option_chain_quality':option_report.quality_score,'candidate_quality':candidate_report.quality_score}
    out = pd.DataFrame([summary])
    path = Path('outputs/institutional_cycle_summary.csv')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        try:
            out = pd.concat([pd.read_csv(path), out], ignore_index=True)
        except Exception:
            pass
    out.to_csv(path, index=False)
    record_event('cycle_recorded', 'Recorded full cycle outputs into institutional database.', payload=str(summary), database_file=database_file)
    print('Institutional recorder complete.')
    print(f"Option chain rows:      {summary['option_chain_rows']}")
    print(f"Raw candidates:         {summary['raw_candidates']}")
    print(f"Filtered candidates:    {summary['filtered_candidates']}")
    print(f"Open positions:         {summary['open_positions']}")
    print(f"Estimated equity:       ${summary['estimated_equity']:,.2f}")
    print(f"Total return:           {summary['total_return']:.2%}")
    return summary


if __name__ == '__main__':
    record_full_cycle_outputs()
