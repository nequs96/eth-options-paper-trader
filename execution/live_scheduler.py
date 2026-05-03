"""
execution/live_scheduler.py

Live paper trading scheduler for ETH options.

This module runs the full paper-trading workflow repeatedly:

1. Refresh live Deribit ETH option chain
2. Scan options for model mispricing
3. Open new paper trades from valid candidates
4. Manage existing paper positions
5. Generate performance report
6. Optionally generate visual reports
7. Sleep and repeat

Important:
This is paper trading only.
It does not place real orders.
It does not use real money.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from pathlib import Path

from backtesting.live_option_backtest_engine import (
    LiveOptionBacktestConfig,
    run_live_option_paper_backtest,
)

from execution.paper_trader import (
    PaperTraderConfig,
    open_paper_trades,
    print_paper_account_summary,
)

from execution.paper_position_manager import manage_paper_positions

from execution.paper_performance_report import generate_paper_performance_report

from visualization.live_option_report import generate_live_option_report


@dataclass
class LiveSchedulerConfig:
    """
    Configuration for live paper trading loop.
    """

    # How often to repeat the full cycle.
    # 1800 seconds = 30 minutes.
    sleep_seconds: int = 1800

    # None = run forever until Ctrl+C.
    # Example: 16 cycles * 30 minutes = about 8 hours.
    max_cycles: int | None = 16

    # Paper trading settings.
    initial_cash: float = 10_000.0
    max_positions: int = 5
    max_risk_per_trade: float = 0.01

    # Scanner settings.
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

    # Exit settings.
    take_profit_pct: float = 0.40
    stop_loss_pct: float = -0.30

    # Reporting settings.
    run_visual_report: bool = True
    show_plots: bool = False

    # Files.
    output_folder: str = "outputs"
    option_chain_file: str = "outputs/live_eth_option_chain.csv"
    candidates_file: str = "outputs/live_backtest_candidates.csv"
    paper_positions_file: str = "outputs/paper_open_positions.csv"
    paper_trade_history_file: str = "outputs/paper_trade_history.csv"
    paper_cash_file: str = "outputs/paper_cash.csv"


def ensure_output_folder(folder: str) -> None:
    """
    Create output folder if it does not exist.
    """

    Path(folder).mkdir(parents=True, exist_ok=True)


def run_scanner_cycle(
    scheduler_config: LiveSchedulerConfig,
) -> None:
    """
    Refresh live option data and generate candidate options.
    """

    print("\n========== STEP 1: LIVE OPTION SCANNER ==========")

    live_backtest_config = LiveOptionBacktestConfig(
        refresh_option_chain=True,
        option_chain_file=scheduler_config.option_chain_file,
        output_folder=scheduler_config.output_folder,
        historical_vol_start_date=scheduler_config.historical_vol_start_date,
        initial_cash=scheduler_config.initial_cash,
        max_risk_per_trade=scheduler_config.max_risk_per_trade,
        max_positions=scheduler_config.max_positions,
        risk_free_rate=scheduler_config.risk_free_rate,
        min_days_to_expiry=scheduler_config.min_days_to_expiry,
        max_days_to_expiry=scheduler_config.max_days_to_expiry,
        min_market_price_usd=scheduler_config.min_market_price_usd,
        max_bid_ask_spread_pct=scheduler_config.max_bid_ask_spread_pct,
        price_threshold=scheduler_config.price_threshold,
        volatility_threshold=scheduler_config.volatility_threshold,
        min_volatility=scheduler_config.min_volatility,
        max_volatility=scheduler_config.max_volatility,
        allow_calls=True,
        allow_puts=True,
        only_trade_cheap_options=True,
    )

    run_live_option_paper_backtest(live_backtest_config)

    print("========== SCANNER FINISHED ==========")


def run_paper_trade_opening_cycle(
    scheduler_config: LiveSchedulerConfig,
) -> PaperTraderConfig:
    """
    Open paper trades from scanner candidates.
    """

    print("\n========== STEP 2: PAPER TRADE OPENER ==========")

    paper_config = PaperTraderConfig(
        candidates_file=scheduler_config.candidates_file,
        positions_file=scheduler_config.paper_positions_file,
        trade_history_file=scheduler_config.paper_trade_history_file,
        initial_cash_file=scheduler_config.paper_cash_file,
        initial_cash=scheduler_config.initial_cash,
        max_positions=scheduler_config.max_positions,
        max_risk_per_trade=scheduler_config.max_risk_per_trade,
        only_trade_cheap_options=True,
        min_combined_score=0.0,
        min_market_price_usd=scheduler_config.min_market_price_usd,
    )

    open_paper_trades(paper_config)

    print("========== PAPER TRADE OPENER FINISHED ==========")

    return paper_config


def run_position_management_cycle(
    paper_config: PaperTraderConfig,
    scheduler_config: LiveSchedulerConfig,
) -> None:
    """
    Manage existing paper positions.
    """

    print("\n========== STEP 3: PAPER POSITION MANAGER ==========")

    manage_paper_positions(
        trader_config=paper_config,
        take_profit_pct=scheduler_config.take_profit_pct,
        stop_loss_pct=scheduler_config.stop_loss_pct,
    )

    print("========== POSITION MANAGER FINISHED ==========")


def run_reporting_cycle(
    paper_config: PaperTraderConfig,
    scheduler_config: LiveSchedulerConfig,
) -> None:
    """
    Generate performance and optional visual reports.
    """

    print("\n========== STEP 4: PAPER PERFORMANCE REPORT ==========")

    generate_paper_performance_report()

    print_paper_account_summary(paper_config)

    if scheduler_config.run_visual_report:
        print("\n========== STEP 5: VISUAL REPORT ==========")

        generate_live_option_report(
            candidates_file=scheduler_config.candidates_file,
            positions_file=scheduler_config.paper_positions_file,
            show_plot=scheduler_config.show_plots,
        )

    print("========== REPORTING FINISHED ==========")


def run_one_cycle(
    scheduler_config: LiveSchedulerConfig,
) -> None:
    """
    Run one full live paper trading cycle.
    """

    cycle_start = time.strftime("%Y-%m-%d %H:%M:%S")

    print("\n====================================================")
    print(f"STARTING LIVE PAPER TRADING CYCLE: {cycle_start}")
    print("====================================================")

    ensure_output_folder(scheduler_config.output_folder)

    run_scanner_cycle(scheduler_config)

    paper_config = run_paper_trade_opening_cycle(scheduler_config)

    run_position_management_cycle(
        paper_config=paper_config,
        scheduler_config=scheduler_config,
    )

    run_reporting_cycle(
        paper_config=paper_config,
        scheduler_config=scheduler_config,
    )

    cycle_end = time.strftime("%Y-%m-%d %H:%M:%S")

    print("\n====================================================")
    print(f"LIVE PAPER TRADING CYCLE FINISHED: {cycle_end}")
    print("====================================================")


def run_live_scheduler(
    scheduler_config: LiveSchedulerConfig | None = None,
) -> None:
    """
    Run live paper trading scheduler.

    The scheduler repeats until:
    - max_cycles is reached, or
    - user presses Ctrl+C.
    """

    if scheduler_config is None:
        scheduler_config = LiveSchedulerConfig()

    ensure_output_folder(scheduler_config.output_folder)

    print("========== LIVE PAPER TRADING SCHEDULER ==========")
    print(f"Sleep seconds:          {scheduler_config.sleep_seconds}")
    print(f"Max cycles:             {scheduler_config.max_cycles}")
    print(f"Initial cash:           ${scheduler_config.initial_cash:,.2f}")
    print(f"Max positions:          {scheduler_config.max_positions}")
    print(f"Max risk per trade:     {scheduler_config.max_risk_per_trade:.2%}")
    print(f"Take profit:            {scheduler_config.take_profit_pct:.2%}")
    print(f"Stop loss:              {scheduler_config.stop_loss_pct:.2%}")
    print("Press Ctrl+C to stop.")
    print("==================================================")

    cycle_number = 0

    while True:
        cycle_number += 1

        print(f"\n\n#################### CYCLE {cycle_number} ####################")

        try:
            run_one_cycle(scheduler_config)

        except KeyboardInterrupt:
            print("\nScheduler stopped by user.")
            break

        except Exception as error:
            print("\nERROR during scheduler cycle:")
            print(error)
            traceback.print_exc()

        if scheduler_config.max_cycles is not None:
            if cycle_number >= scheduler_config.max_cycles:
                print("\nReached max_cycles. Scheduler finished.")
                break

        print(
            f"\nSleeping for {scheduler_config.sleep_seconds} seconds "
            "before next cycle..."
        )

        try:
            time.sleep(scheduler_config.sleep_seconds)
        except KeyboardInterrupt:
            print("\nScheduler stopped by user during sleep.")
            break


if __name__ == "__main__":
    # Recommended first overnight test:
    # 16 cycles × 30 minutes = about 8 hours.

    config = LiveSchedulerConfig(
        sleep_seconds=1800,
        max_cycles=16,
        initial_cash=10_000.0,
        max_positions=5,
        max_risk_per_trade=0.01,
        historical_vol_start_date="2023-01-01",
        risk_free_rate=0.04,
        min_days_to_expiry=3.0,
        max_days_to_expiry=45.0,
        min_market_price_usd=5.0,
        max_bid_ask_spread_pct=0.35,
        price_threshold=0.10,
        volatility_threshold=0.10,
        min_volatility=0.10,
        max_volatility=2.50,
        take_profit_pct=0.40,
        stop_loss_pct=-0.30,
        run_visual_report=True,
        show_plots=False,
        output_folder="outputs",
        option_chain_file="outputs/live_eth_option_chain.csv",
        candidates_file="outputs/live_backtest_candidates.csv",
        paper_positions_file="outputs/paper_open_positions.csv",
        paper_trade_history_file="outputs/paper_trade_history.csv",
        paper_cash_file="outputs/paper_cash.csv",
    )

    run_live_scheduler(config)