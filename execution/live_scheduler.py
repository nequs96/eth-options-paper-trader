from __future__ import annotations
import dataclasses
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from backtesting.live_option_backtest_engine import LiveOptionBacktestConfig, run_live_option_paper_backtest
from execution.professional_candidate_filter import ProfessionalFilterConfig, filter_candidate_file
from execution.paper_trader import PaperTraderConfig, open_paper_trades
from execution.paper_position_manager import manage_paper_positions
from execution.paper_account_reconciliation import generate_reconciliation_report
from execution.paper_performance_report import generate_paper_performance_report, print_performance_report
from execution.paper_equity_curve import generate_equity_curve
from execution.paper_risk_metrics import calculate_risk_metrics, print_risk_metrics_report
from execution.runtime_healthcheck import run_runtime_healthcheck
from execution.clean_reporting import print_clean_cycle_report


@dataclass
class LiveSchedulerConfig:
    output_folder: str = 'outputs'
    sleep_seconds: int = 900
    max_cycles: int | None = 1
    option_chain_file: str = 'outputs/live_eth_option_chain.csv'
    raw_candidates_file: str = 'outputs/live_backtest_candidates.csv'
    filtered_candidates_file: str = 'outputs/live_backtest_candidates_filtered.csv'
    rejected_candidates_file: str = 'outputs/live_backtest_candidates_rejected.csv'
    paper_cash_file: str = 'outputs/paper_cash.csv'
    paper_positions_file: str = 'outputs/paper_open_positions.csv'
    paper_trade_history_file: str = 'outputs/paper_trade_history.csv'
    refresh_option_chain: bool = True
    historical_vol_start_date: str = '2023-01-01'
    initial_cash: float = 10000.0
    risk_free_rate: float = 0.04
    min_days_to_expiry: float = 3.0
    max_days_to_expiry: float = 45.0
    min_market_price_usd: float = 5.0
    max_bid_ask_spread_pct: float = 0.35
    price_threshold: float = 0.10
    volatility_threshold: float = 0.10
    min_volatility: float = 0.10
    max_volatility: float = 2.50
    allow_calls: bool = True
    allow_puts: bool = True
    only_trade_cheap_options: bool = True
    max_positions: int = 8
    target_positions: int = 3
    max_new_positions_per_cycle: int = 2
    normal_min_score: float = 0.30
    expansion_min_score: float = 0.50
    exceptional_min_score: float = 0.70
    min_relative_to_best_score: float = 0.80
    min_risk_per_trade: float = 0.001
    normal_max_risk_per_trade: float = 0.0125
    exceptional_max_risk_per_trade: float = 0.020
    max_total_open_risk_pct: float = 0.10
    run_healthcheck: bool = True
    strict_healthcheck: bool = True
    max_option_chain_instruments: int | None = 150
    ticker_timeout_seconds: int = 5
    option_chain_progress_every: int = 25
    save_partial_option_chain: bool = True
    fallback_historical_volatility: float = 0.75
    clean_terminal_report: bool = True


def build_dataclass_config(cls: type, values: dict):
    allowed = {field.name for field in dataclasses.fields(cls)}
    return cls(**{key: value for key, value in values.items() if key in allowed})


def build_scanner_config(cfg: LiveSchedulerConfig) -> LiveOptionBacktestConfig:
    values = dict(cfg.__dict__)
    values['max_risk_per_trade'] = cfg.normal_max_risk_per_trade
    return build_dataclass_config(LiveOptionBacktestConfig, values)


def build_filter_config(cfg: LiveSchedulerConfig) -> ProfessionalFilterConfig:
    return build_dataclass_config(ProfessionalFilterConfig, {'raw_candidates_file': cfg.raw_candidates_file, 'filtered_candidates_file': cfg.filtered_candidates_file, 'rejected_candidates_file': cfg.rejected_candidates_file, 'historical_vol_start_date': cfg.historical_vol_start_date, 'min_days_to_expiry': cfg.min_days_to_expiry, 'max_days_to_expiry': cfg.max_days_to_expiry, 'min_market_price_usd': cfg.min_market_price_usd, 'max_bid_ask_spread_pct': cfg.max_bid_ask_spread_pct})


def build_paper_config(cfg: LiveSchedulerConfig) -> PaperTraderConfig:
    return build_dataclass_config(PaperTraderConfig, {'candidates_file': cfg.filtered_candidates_file, 'positions_file': cfg.paper_positions_file, 'trade_history_file': cfg.paper_trade_history_file, 'initial_cash_file': cfg.paper_cash_file, 'initial_cash': cfg.initial_cash, 'min_risk_per_trade': cfg.min_risk_per_trade, 'normal_max_risk_per_trade': cfg.normal_max_risk_per_trade, 'exceptional_max_risk_per_trade': cfg.exceptional_max_risk_per_trade, 'max_total_open_risk_pct': cfg.max_total_open_risk_pct, 'max_positions': cfg.max_positions, 'target_positions': cfg.target_positions, 'max_new_positions_per_cycle': cfg.max_new_positions_per_cycle, 'normal_min_score': cfg.normal_min_score, 'expansion_min_score': cfg.expansion_min_score, 'exceptional_min_score': cfg.exceptional_min_score, 'min_relative_to_best_score': cfg.min_relative_to_best_score, 'only_trade_cheap_options': cfg.only_trade_cheap_options, 'min_market_price_usd': cfg.min_market_price_usd})


def reconciliation_is_ok(paper_cfg: PaperTraderConfig) -> bool:
    report = generate_reconciliation_report(paper_cfg)
    return bool(not report.empty and report.iloc[-1].get('reconciliation_ok', False))


def print_passed_options_report(filtered, top_n: int = 15) -> None:
    print('\n========== OPTIONS THAT PASSED FILTER =========')
    if filtered is None or filtered.empty:
        print('No options passed current-cycle professional filter.')
        return
    data = filtered.copy()
    if 'mci' in data.columns:
        data = data.sort_values('mci', ascending=False)
    useful = ['instrument_name','option_type','strike','days_to_expiry','market_price_usd','model_price_usd','price_diff_pct','volatility_spread','mci','decision_reason']
    existing = [c for c in useful if c in data.columns]
    print(f'Passed options: {len(data)}')
    print(data[existing].head(top_n).to_string(index=False))


def run_one_cycle(cfg: LiveSchedulerConfig) -> None:
    Path(cfg.output_folder).mkdir(parents=True, exist_ok=True)
    paper_cfg = build_paper_config(cfg)
    print('\n====================================================')
    print(f"STARTING PAPER CYCLE: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print('====================================================')
    if cfg.run_healthcheck:
        run_runtime_healthcheck(cfg.output_folder, paper_cfg, strict=cfg.strict_healthcheck)
    print('\n========== STEP 1: MANAGE OPEN POSITIONS =========')
    manage_paper_positions(paper_cfg)
    print('\n========== STEP 2: RECONCILE BEFORE NEW RISK =========')
    if not reconciliation_is_ok(paper_cfg):
        print('Reconciliation hard-failed. Skipping new risk for this cycle.')
        return
    print('\n========== STEP 3: BUILD RAW CANDIDATES =========')
    run_live_option_paper_backtest(build_scanner_config(cfg))
    print('\n========== STEP 4: PROFESSIONAL DYNAMIC FILTER =========')
    filtered = filter_candidate_file(build_filter_config(cfg))
    print_passed_options_report(filtered)
    print('\n========== STEP 5: DYNAMIC ALLOCATION + OPEN =========')
    open_paper_trades(paper_cfg)
    print('\n========== STEP 6: RECONCILE AFTER OPENING =========')
    reconciliation_is_ok(paper_cfg)
    print('\n========== STEP 7: REPORTING =========')
    performance = generate_paper_performance_report(paper_cfg)
    generate_equity_curve(paper_cfg)
    risk = calculate_risk_metrics(config=paper_cfg)
    if cfg.clean_terminal_report:
        print_clean_cycle_report(paper_cfg)
    print_performance_report(performance)
    print_risk_metrics_report(risk)


def run_live_scheduler(cfg: LiveSchedulerConfig | None = None) -> None:
    cfg = cfg or LiveSchedulerConfig()
    cycle = 0
    while True:
        cycle += 1
        try:
            run_one_cycle(cfg)
        except KeyboardInterrupt:
            print('\nScheduler stopped by user.')
            break
        except Exception:
            print('ERROR during scheduler cycle:')
            traceback.print_exc()
        if cfg.max_cycles is not None and cycle >= cfg.max_cycles:
            print(f'Reached max_cycles={cfg.max_cycles}. Stopping scheduler.')
            break
        try:
            print(f'Sleeping {cfg.sleep_seconds} seconds...')
            time.sleep(cfg.sleep_seconds)
        except KeyboardInterrupt:
            print('\nScheduler stopped by user during sleep.')
            break


if __name__ == '__main__':
    run_live_scheduler(LiveSchedulerConfig(sleep_seconds=900, max_cycles=None))
