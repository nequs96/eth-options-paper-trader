"""
execution/live_scheduler.py

Professional live paper trading scheduler for ETH options.

Workflow:
1. Refresh live Deribit ETH option chain
2. Generate raw model-mispricing candidates
3. Apply professional candidate filter
4. Manage existing paper positions
5. Reconcile account before opening new trades
6. Open new paper trades from filtered candidates
7. Generate reports, equity curve, risk metrics, reconciliation
8. Sleep and repeat

Important:
This is paper trading only.
No real orders are placed.
Profitability is not guaranteed.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from pathlib import Path

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
from execution.paper_performance_report import generate_paper_performance_report
from execution.paper_equity_curve import generate_equity_curve
from execution.paper_risk_metrics import calculate_risk_metrics
from execution.paper_account_reconciliation import generate_reconciliation_report
from visualization.live_option_report import generate_live_option_report


@dataclass
class LiveSchedulerConfig:
    """
    Professional live paper scheduler configuration.
    """

    # Loop control
    sleep_seconds: int = 1000
    max_cycles: int | None = 1

    # Paper trading
    initial_cash: float = 10_000.0
    max_positions: int = 3
    max_risk_per_trade: float = 0.01

    # Raw scanner settings
    historical_vol_start_date: str = "2023-01-01"
    risk_free_rate: float = 0.04
    min_days_to_expiry: float = 10.0
    max_days_to_expiry: float = 30.0
    min_market_price_usd: float = 10.0
    max_bid_ask_spread_pct: float = 0.25
    price_threshold: float = 0.10
    volatility_threshold: float = 0.10
    min_volatility: float = 0.10
    max_volatility: float = 2.50
    only_trade_cheap_options: bool = True

    # Professional post-filter settings
    min_combined_score: float = 0.15
    max_price_diff_pct: float = -0.12
    max_volatility_spread: float = -0.08
    professional_max_bid_ask_spread_pct: float = 0.20
    min_abs_delta: float = 0.25
    max_abs_delta: float = 0.65

    # Exit rules
    take_profit_pct: float = 0.35
    stop_loss_pct: float = -0.20
    min_days_to_expiry_exit: float = 1.0

    # Reports
    run_visual_report: bool = False
    show_plots: bool = False

    # Files
    output_folder: str = "outputs"
    option_chain_file: str = "outputs/live_eth_option_chain.csv"
    raw_candidates_file: str = "outputs/live_backtest_candidates.csv"
    filtered_candidates_file: str = "outputs/live_backtest_candidates_filtered.csv"
    rejected_candidates_file: str = "outputs/live_backtest_candidates_rejected.csv"
    paper_positions_file: str = "outputs/paper_open_positions.csv"
    paper_trade_history_file: str = "outputs/paper_trade_history.csv"
    paper_cash_file: str = "outputs/paper_cash.csv"


def ensure_output_folder(folder: str) -> None:
    Path(folder).mkdir(parents=True, exist_ok=True)


def file_exists_and_not_empty(file_path: str) -> bool:
    path = Path(file_path)
    return path.exists() and path.stat().st_size > 0


def csv_has_rows(file_path: str) -> bool:
    if not file_exists_and_not_empty(file_path):
        return False

    try:
        data = pd.read_csv(file_path, nrows=1)
    except Exception:
        return False

    return not data.empty


def run_scanner_cycle(cfg: LiveSchedulerConfig) -> None:
    """
    Run raw live option candidate generation.

    This writes:
    - outputs/live_backtest_candidates.csv
    """

    print("\n========== STEP 1: RAW OPTION SCANNER ==========")

    scan_cfg = LiveOptionBacktestConfig(
        refresh_option_chain=True,
        option_chain_file=cfg.option_chain_file,
        output_folder=cfg.output_folder,
        historical_vol_start_date=cfg.historical_vol_start_date,
        initial_cash=cfg.initial_cash,
        max_risk_per_trade=cfg.max_risk_per_trade,
        max_positions=cfg.max_positions,
        risk_free_rate=cfg.risk_free_rate,
        min_days_to_expiry=cfg.min_days_to_expiry,
        max_days_to_expiry=cfg.max_days_to_expiry,
        min_market_price_usd=cfg.min_market_price_usd,
        max_bid_ask_spread_pct=cfg.max_bid_ask_spread_pct,
        price_threshold=cfg.price_threshold,
        volatility_threshold=cfg.volatility_threshold,
        min_volatility=cfg.min_volatility,
        max_volatility=cfg.max_volatility,
        allow_calls=True,
        allow_puts=True,
        only_trade_cheap_options=cfg.only_trade_cheap_options,
    )

    run_live_option_paper_backtest(scan_cfg)

    print("========== RAW SCANNER DONE ==========")


def run_professional_filter_cycle(cfg: LiveSchedulerConfig) -> pd.DataFrame:
    """
    Apply professional filters to raw candidates.

    This writes:
    - outputs/live_backtest_candidates_filtered.csv
    - outputs/live_backtest_candidates_rejected.csv
    """

    print("\n========== STEP 2: PROFESSIONAL CANDIDATE FILTER ==========")

    filter_cfg = ProfessionalFilterConfig(
        raw_candidates_file=cfg.raw_candidates_file,
        filtered_candidates_file=cfg.filtered_candidates_file,
        rejected_candidates_file=cfg.rejected_candidates_file,
        historical_vol_start_date=cfg.historical_vol_start_date,
        only_trade_cheap_options=cfg.only_trade_cheap_options,
        min_combined_score=cfg.min_combined_score,
        max_price_diff_pct=cfg.max_price_diff_pct,
        max_volatility_spread=cfg.max_volatility_spread,
        min_market_price_usd=cfg.min_market_price_usd,
        max_bid_ask_spread_pct=cfg.professional_max_bid_ask_spread_pct,
        min_days_to_expiry=cfg.min_days_to_expiry,
        max_days_to_expiry=cfg.max_days_to_expiry,
        min_abs_delta=cfg.min_abs_delta,
        max_abs_delta=cfg.max_abs_delta,
    )

    filtered = filter_candidate_file(filter_cfg)

    print("========== PROFESSIONAL FILTER DONE ==========")

    return filtered


def build_paper_config(cfg: LiveSchedulerConfig) -> PaperTraderConfig:
    """
    Build persistent paper trader config.

    Important:
    paper_trader reads the FILTERED candidate file, not the raw candidate file.
    """

    return PaperTraderConfig(
        candidates_file=cfg.filtered_candidates_file,
        positions_file=cfg.paper_positions_file,
        trade_history_file=cfg.paper_trade_history_file,
        initial_cash_file=cfg.paper_cash_file,
        initial_cash=cfg.initial_cash,
        max_positions=cfg.max_positions,
        max_risk_per_trade=cfg.max_risk_per_trade,
        only_trade_cheap_options=cfg.only_trade_cheap_options,
        min_abs_mispricing_score=cfg.min_combined_score,
        min_market_price_usd=cfg.min_market_price_usd,
    )


def reconciliation_is_ok() -> bool:
    """
    Run reconciliation and return True only if account state is clean.
    """

    report = generate_reconciliation_report()

    if report.empty or "reconciliation_ok" not in report.columns:
        return False

    value = report["reconciliation_ok"].iloc[0]

    return bool(value)


def run_position_management_cycle(
    paper_cfg: PaperTraderConfig,
    cfg: LiveSchedulerConfig,
) -> None:
    print("\n========== STEP 3: POSITION MANAGEMENT ==========")

    manage_paper_positions(
        trader_config=paper_cfg,
        take_profit_pct=cfg.take_profit_pct,
        stop_loss_pct=cfg.stop_loss_pct,
        min_days_to_expiry=cfg.min_days_to_expiry_exit,
    )

    print("========== POSITION MANAGEMENT DONE ==========")


def run_trade_opening_cycle(
    paper_cfg: PaperTraderConfig,
    cfg: LiveSchedulerConfig,
) -> None:
    print("\n========== STEP 4: OPEN PAPER TRADES ==========")

    if not csv_has_rows(cfg.filtered_candidates_file):
        print(
            "No filtered candidates available. "
            "Skipping new paper trade entries."
        )
        return

    if not reconciliation_is_ok():
        print(
            "WARNING: Account reconciliation failed. "
            "Skipping new trade entries to prevent corrupted accounting."
        )
        return

    open_paper_trades(paper_cfg)

    print("========== PAPER TRADE OPENING DONE ==========")


def run_reporting_cycle(
    paper_cfg: PaperTraderConfig,
    cfg: LiveSchedulerConfig,
) -> None:
    print("\n========== STEP 5: REPORTING ==========")

    try:
        generate_paper_performance_report()
    except Exception:
        print("WARNING: Performance report failed.")
        traceback.print_exc()

    try:
        print_paper_account_summary(paper_cfg)
    except Exception:
        print("WARNING: Paper account summary failed.")
        traceback.print_exc()

    try:
        generate_equity_curve()
    except Exception:
        print("WARNING: Equity curve generation failed.")
        traceback.print_exc()

    try:
        calculate_risk_metrics()
    except Exception:
        print("WARNING: Risk metrics calculation failed.")
        traceback.print_exc()

    try:
        generate_reconciliation_report()
    except Exception:
        print("WARNING: Final reconciliation failed.")
        traceback.print_exc()

    if cfg.run_visual_report:
        try:
            generate_live_option_report(
                candidates_file=cfg.filtered_candidates_file,
                positions_file=cfg.paper_positions_file,
                show_plot=cfg.show_plots,
            )
        except Exception:
            print("WARNING: Visual report generation failed.")
            traceback.print_exc()

    print("========== REPORTING DONE ==========")


def run_one_cycle(cfg: LiveSchedulerConfig) -> None:
    print("\n====================================================")
    print(f"STARTING PROFESSIONAL PAPER CYCLE: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("====================================================")

    ensure_output_folder(cfg.output_folder)

    paper_cfg = build_paper_config(cfg)

    # 1. Generate raw candidates
    try:
        run_scanner_cycle(cfg)
    except Exception:
        print("WARNING: Raw scanner failed.")
        traceback.print_exc()

    # 2. Apply professional candidate filter
    try:
        run_professional_filter_cycle(cfg)
    except Exception:
        print("WARNING: Professional candidate filter failed.")
        traceback.print_exc()

    # 3. Manage old positions before opening new positions
    try:
        run_position_management_cycle(
            paper_cfg=paper_cfg,
            cfg=cfg,
        )
    except Exception:
        print("WARNING: Position management failed.")
        traceback.print_exc()

    # 4. Open new paper trades only if account reconciles
    try:
        run_trade_opening_cycle(
            paper_cfg=paper_cfg,
            cfg=cfg,
        )
    except Exception:
        print("WARNING: Trade opening failed.")
        traceback.print_exc()

    # 5. Reports
    try:
        run_reporting_cycle(
            paper_cfg=paper_cfg,
            cfg=cfg,
        )
    except Exception:
        print("WARNING: Reporting cycle failed.")
        traceback.print_exc()

    print("\n====================================================")
    print(f"CYCLE FINISHED: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("====================================================")


def run_live_scheduler(cfg: LiveSchedulerConfig | None = None) -> None:
    if cfg is None:
        cfg = LiveSchedulerConfig()

    ensure_output_folder(cfg.output_folder)

    print("\n========== PROFESSIONAL ETH OPTIONS PAPER SCHEDULER ==========")
    print(f"Initial cash:                 ${cfg.initial_cash:,.2f}")
    print(f"Max positions:                {cfg.max_positions}")
    print(f"Risk per trade:               {cfg.max_risk_per_trade:.2%}")
    print(f"Raw price threshold:          {cfg.price_threshold:.2%}")
    print(f"Raw volatility threshold:     {cfg.volatility_threshold:.2%}")
    print(f"Professional min score:       {cfg.min_combined_score:.4f}")
    print(f"Max spread:                   {cfg.professional_max_bid_ask_spread_pct:.2%}")
    print(f"DTE window:                   {cfg.min_days_to_expiry} - {cfg.max_days_to_expiry}")
    print(f"Take profit:                  {cfg.take_profit_pct:.2%}")
    print(f"Stop loss:                    {cfg.stop_loss_pct:.2%}")
    print(f"Sleep seconds:                {cfg.sleep_seconds}")
    print(f"Max cycles:                   {cfg.max_cycles}")
    print("===============================================================")

    cycle = 0

    while True:
        cycle += 1

        print(f"\n#################### CYCLE {cycle} ####################")

        try:
            run_one_cycle(cfg)
        except KeyboardInterrupt:
            print("Scheduler stopped by user.")
            break
        except Exception:
            print("FATAL ERROR in scheduler loop.")
            traceback.print_exc()

        if cfg.max_cycles is not None and cycle >= cfg.max_cycles:
            print("Reached max_cycles. Scheduler exiting.")
            break

        try:
            print(f"Sleeping for {cfg.sleep_seconds} seconds...")
            time.sleep(cfg.sleep_seconds)
        except KeyboardInterrupt:
            print("Scheduler stopped during sleep.")
            break


if __name__ == "__main__":
    # First safe test: one cycle only.
    config = LiveSchedulerConfig(
        sleep_seconds=1000,
        max_cycles=200,

        # Professional stricter settings
        initial_cash=10_000.0,
        max_positions=10,
        max_risk_per_trade=0.01,

        min_days_to_expiry=10.0,
        max_days_to_expiry=30.0,
        min_market_price_usd=10.0,
        max_bid_ask_spread_pct=0.25,

        price_threshold=0.10,
        volatility_threshold=0.10,

        min_combined_score=0.15,
        max_price_diff_pct=-0.12,
        max_volatility_spread=-0.08,
        professional_max_bid_ask_spread_pct=0.20,

        take_profit_pct=0.35,
        stop_loss_pct=-0.20,

        run_visual_report=False,
        show_plots=False,
    )

    run_live_scheduler(config)