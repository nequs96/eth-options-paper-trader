"""
execution/live_scheduler.py

Cleaner paper-trading scheduler.

Target cycle:
1. Manage existing positions first.
2. Reconcile account before adding new risk.
3. Build raw candidates only.
4. Apply professional candidate filter.
5. Open new paper trades through dynamic portfolio allocation.
6. Reconcile again.
7. Generate lightweight reports.
8. Optional visual report.

Paper trading only. No real orders are placed.
"""

from __future__ import annotations

import dataclasses
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from backtesting.live_option_backtest_engine import (
    LiveOptionBacktestConfig,
    run_live_option_paper_backtest,
)
from execution.professional_candidate_filter import (
    ProfessionalFilterConfig,
    filter_candidate_file,
)
from execution.paper_trader import (
    PaperTraderConfig,
    open_paper_trades,
    print_paper_account_summary,
)
from execution.paper_position_manager import manage_paper_positions
from execution.paper_account_reconciliation import generate_reconciliation_report
from execution.paper_performance_report import generate_paper_performance_report
from execution.paper_equity_curve import generate_equity_curve
from execution.paper_risk_metrics import calculate_risk_metrics


@dataclass
class LiveSchedulerConfig:
    output_folder: str = "outputs"
    sleep_seconds: int = 1000
    max_cycles: int | None = 1

    refresh_option_chain: bool = True
    option_chain_file: str = "outputs/live_eth_option_chain.csv"
    raw_candidates_file: str = "outputs/live_backtest_candidates.csv"
    filtered_candidates_file: str = "outputs/live_backtest_candidates_filtered.csv"
    rejected_candidates_file: str = "outputs/live_backtest_candidates_rejected.csv"

    paper_cash_file: str = "outputs/paper_cash.csv"
    paper_positions_file: str = "outputs/paper_open_positions.csv"
    paper_trade_history_file: str = "outputs/paper_trade_history.csv"

    initial_cash: float = 10_000.0
    max_risk_per_trade: float = 0.01

    # Dynamic allocation settings.
    max_positions: int = 30
    target_positions: int = 4
    max_new_positions_per_cycle: int = 2
    normal_min_score: float = 0.25
    expansion_min_score: float = 0.45
    exceptional_min_score: float = 0.60
    min_relative_to_best_score: float = 0.75

    # Exit settings.
    take_profit_pct: float = 0.30
    stop_loss_pct: float = -0.25
    min_days_to_expiry_exit: float = 1.5

    # Candidate generation/filter settings.
    historical_vol_start_date: str = "2023-01-01"
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

    # Optional expensive reporting.
    run_visual_report: bool = False


def ensure_output_folder(folder: str) -> None:
    Path(folder).mkdir(parents=True, exist_ok=True)


def csv_has_rows(file_path: str) -> bool:
    path = Path(file_path)
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        return not pd.read_csv(path).empty
    except Exception:
        return False


def build_dataclass_config(cls: type, values: dict[str, Any]) -> Any:
    """
    Build external config dataclasses safely.
    This lets the scheduler survive small differences between config versions.
    """
    if not dataclasses.is_dataclass(cls):
        return cls()
    allowed = {field.name for field in dataclasses.fields(cls)}
    kwargs = {key: value for key, value in values.items() if key in allowed}
    return cls(**kwargs)


def build_scanner_config(cfg: LiveSchedulerConfig) -> LiveOptionBacktestConfig:
    return build_dataclass_config(
        LiveOptionBacktestConfig,
        {
            "refresh_option_chain": cfg.refresh_option_chain,
            "option_chain_file": cfg.option_chain_file,
            "output_folder": cfg.output_folder,
            "historical_vol_start_date": cfg.historical_vol_start_date,
            "initial_cash": cfg.initial_cash,
            "max_risk_per_trade": cfg.max_risk_per_trade,
            "max_positions": cfg.max_positions,
            "risk_free_rate": cfg.risk_free_rate,
            "min_days_to_expiry": cfg.min_days_to_expiry,
            "max_days_to_expiry": cfg.max_days_to_expiry,
            "min_market_price_usd": cfg.min_market_price_usd,
            "max_bid_ask_spread_pct": cfg.max_bid_ask_spread_pct,
            "price_threshold": cfg.price_threshold,
            "volatility_threshold": cfg.volatility_threshold,
            "min_volatility": cfg.min_volatility,
            "max_volatility": cfg.max_volatility,
            "allow_calls": cfg.allow_calls,
            "allow_puts": cfg.allow_puts,
            "only_trade_cheap_options": cfg.only_trade_cheap_options,
        },
    )


def build_filter_config(cfg: LiveSchedulerConfig) -> ProfessionalFilterConfig:
    return build_dataclass_config(
        ProfessionalFilterConfig,
        {
            "raw_candidates_file": cfg.raw_candidates_file,
            "filtered_candidates_file": cfg.filtered_candidates_file,
            "rejected_candidates_file": cfg.rejected_candidates_file,
            "historical_vol_start_date": cfg.historical_vol_start_date,
            "min_days_to_expiry": cfg.min_days_to_expiry,
            "max_days_to_expiry": cfg.max_days_to_expiry,
            "min_market_price_usd": cfg.min_market_price_usd,
            "max_bid_ask_spread_pct": cfg.max_bid_ask_spread_pct,
        },
    )


def build_paper_config(cfg: LiveSchedulerConfig) -> PaperTraderConfig:
    return PaperTraderConfig(
        candidates_file=cfg.filtered_candidates_file,
        positions_file=cfg.paper_positions_file,
        trade_history_file=cfg.paper_trade_history_file,
        initial_cash_file=cfg.paper_cash_file,
        initial_cash=cfg.initial_cash,
        max_risk_per_trade=cfg.max_risk_per_trade,
        max_positions=cfg.max_positions,
        target_positions=cfg.target_positions,
        max_new_positions_per_cycle=cfg.max_new_positions_per_cycle,
        normal_min_score=cfg.normal_min_score,
        expansion_min_score=cfg.expansion_min_score,
        exceptional_min_score=cfg.exceptional_min_score,
        min_relative_to_best_score=cfg.min_relative_to_best_score,
        only_trade_cheap_options=cfg.only_trade_cheap_options,
        min_market_price_usd=cfg.min_market_price_usd,
    )


def reconciliation_is_ok(paper_cfg: PaperTraderConfig) -> bool:
    report = generate_reconciliation_report(paper_cfg)
    if report.empty or "reconciliation_ok" not in report.columns:
        return False
    return bool(report.iloc[-1]["reconciliation_ok"])


def run_position_management_cycle(paper_cfg: PaperTraderConfig, cfg: LiveSchedulerConfig) -> None:
    print("\n========== STEP 1: MANAGE OPEN POSITIONS ==========")
    manage_paper_positions(
        trader_config=paper_cfg,
        take_profit_pct=cfg.take_profit_pct,
        stop_loss_pct=cfg.stop_loss_pct,
        min_days_to_expiry=cfg.min_days_to_expiry_exit,
    )


def run_scanner_cycle(cfg: LiveSchedulerConfig) -> None:
    print("\n========== STEP 3: BUILD RAW CANDIDATES ==========")
    scanner_cfg = build_scanner_config(cfg)
    run_live_option_paper_backtest(scanner_cfg)


def run_professional_filter_cycle(cfg: LiveSchedulerConfig) -> pd.DataFrame:
    print("\n========== STEP 4: PROFESSIONAL FILTER ==========")
    filter_cfg = build_filter_config(cfg)
    filtered = filter_candidate_file(filter_cfg)
    if filtered is None:
        return pd.DataFrame()
    return filtered


def run_trade_opening_cycle(paper_cfg: PaperTraderConfig) -> None:
    print("\n========== STEP 5: DYNAMIC ALLOCATION + OPEN ==========")
    open_paper_trades(paper_cfg)


def run_reporting_cycle(paper_cfg: PaperTraderConfig, cfg: LiveSchedulerConfig) -> None:
    print("\n========== STEP 7: REPORTING ==========")
    generate_paper_performance_report(paper_cfg)
    generate_equity_curve(paper_cfg)
    calculate_risk_metrics()
    print_paper_account_summary(paper_cfg)

    if cfg.run_visual_report:
        try:
            from visualization.live_option_report import generate_live_option_report

            generate_live_option_report(
                candidates_file=cfg.raw_candidates_file,
                positions_file=cfg.paper_positions_file,
                show_plot=False,
            )
        except Exception as error:
            print(f"Visual report skipped: {error}")


def run_one_cycle(cfg: LiveSchedulerConfig) -> None:
    ensure_output_folder(cfg.output_folder)
    paper_cfg = build_paper_config(cfg)

    print("\n====================================================")
    print(f"STARTING PAPER CYCLE: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("====================================================")

    run_position_management_cycle(paper_cfg, cfg)

    print("\n========== STEP 2: RECONCILE BEFORE NEW RISK ==========")
    if not reconciliation_is_ok(paper_cfg):
        print("Reconciliation failed. Skipping scanner/opening/reporting for safety.")
        return

    run_scanner_cycle(cfg)
    filtered = run_professional_filter_cycle(cfg)

    if filtered.empty and not csv_has_rows(cfg.filtered_candidates_file):
        print("No filtered candidates. Skipping trade opening.")
    else:
        run_trade_opening_cycle(paper_cfg)

    print("\n========== STEP 6: RECONCILE AFTER OPENING ==========")
    reconciliation_is_ok(paper_cfg)

    run_reporting_cycle(paper_cfg, cfg)


def run_live_scheduler(cfg: LiveSchedulerConfig | None = None) -> None:
    if cfg is None:
        cfg = LiveSchedulerConfig()

    cycle = 0
    while True:
        cycle += 1
        try:
            run_one_cycle(cfg)
        except KeyboardInterrupt:
            print("Scheduler stopped by user.")
            break
        except Exception:
            print("ERROR during scheduler cycle:")
            traceback.print_exc()

        if cfg.max_cycles is not None and cycle >= cfg.max_cycles:
            print(f"Reached max_cycles={cfg.max_cycles}. Stopping scheduler.")
            break

        print(f"Sleeping {cfg.sleep_seconds} seconds...")
        time.sleep(cfg.sleep_seconds)


if __name__ == "__main__":
    config = LiveSchedulerConfig(
        sleep_seconds=800,
        max_cycles=200,
        max_positions=30,
        target_positions=4,
        max_new_positions_per_cycle=2,
        run_visual_report=False,
    )
    run_live_scheduler(config)
