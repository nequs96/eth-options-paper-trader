"""
execution/strategy_tuning.py

Runs isolated parameter tests for the paper trading strategy.

Each parameter combination gets its own output folder, so results do not
pollute the main paper account.
"""

from pathlib import Path
import itertools

import pandas as pd

from execution.live_scheduler import (
    LiveSchedulerConfig,
    run_scanner_cycle,
    build_paper_config,
)
from execution.paper_position_manager import manage_paper_positions
from execution.paper_trader import open_paper_trades


OUTPUT_FILE = "outputs/strategy_tuning_results.csv"
TUNING_FOLDER = "outputs/tuning_runs"

PRICE_THRESHOLDS = [0.05, 0.10, 0.15]
VOL_THRESHOLDS = [0.05, 0.10]
TAKE_PROFITS = [0.30, 0.40, 0.50]
STOP_LOSSES = [-0.20, -0.30, -0.40]

INITIAL_CASH = 10_000.0


def load_csv_if_exists(path: str) -> pd.DataFrame:
    p = Path(path)

    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()

    try:
        return pd.read_csv(p)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def numeric_column(
    df: pd.DataFrame,
    possible_columns: list[str],
    default: float = 0.0,
) -> pd.Series:
    for col in possible_columns:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(default)

    return pd.Series([default] * len(df), index=df.index)


def summarize_run(run_folder: Path) -> dict:
    cash_file = run_folder / "paper_cash.csv"
    positions_file = run_folder / "paper_open_positions.csv"
    history_file = run_folder / "paper_trade_history.csv"
    candidates_file = run_folder / "live_backtest_candidates.csv"

    cash_df = load_csv_if_exists(str(cash_file))
    positions = load_csv_if_exists(str(positions_file))
    history = load_csv_if_exists(str(history_file))
    candidates = load_csv_if_exists(str(candidates_file))

    if not cash_df.empty and "cash" in cash_df.columns:
        cash = float(pd.to_numeric(cash_df["cash"], errors="coerce").dropna().iloc[0])
    else:
        cash = INITIAL_CASH

    open_current_value = 0.0
    open_unrealized_pnl = 0.0

    if not positions.empty:
        if "current_value_usd" in positions.columns:
            open_current_value = float(
                pd.to_numeric(
                    positions["current_value_usd"],
                    errors="coerce",
                ).fillna(0.0).sum()
            )
        else:
            capital = numeric_column(positions, ["capital_at_risk"])
            unrealized = numeric_column(positions, ["unrealized_pnl_usd", "unrealized_pnl"])
            open_current_value = float((capital + unrealized).sum())

        open_unrealized_pnl = float(
            numeric_column(
                positions,
                ["unrealized_pnl_usd", "unrealized_pnl"],
            ).sum()
        )

    closed_trades = 0
    total_realized_pnl = 0.0

    if not history.empty:
        closed = history.copy()

        if "status" in closed.columns:
            closed = closed[closed["status"].astype(str).str.lower() == "closed"].copy()

        if not closed.empty:
            pnl = numeric_column(closed, ["pnl_usd", "pnl"])
            closed_trades = len(closed)
            total_realized_pnl = float(pnl.sum())

    equity = cash + open_current_value

    cheap_candidates = 0
    total_candidates = len(candidates)

    if not candidates.empty and "classification" in candidates.columns:
        cheap_candidates = int(
            (candidates["classification"].astype(str).str.lower() == "cheap").sum()
        )

    return {
        "cash": cash,
        "equity": equity,
        "open_positions": int(len(positions)),
        "open_current_value": open_current_value,
        "open_unrealized_pnl": open_unrealized_pnl,
        "closed_trades": int(closed_trades),
        "total_realized_pnl": total_realized_pnl,
        "total_candidates": int(total_candidates),
        "cheap_candidates": int(cheap_candidates),
        "total_return_pct": (equity - INITIAL_CASH) / INITIAL_CASH,
    }


def run_strategy_tuning() -> pd.DataFrame:
    Path(TUNING_FOLDER).mkdir(parents=True, exist_ok=True)

    results = []

    combinations = itertools.product(
        PRICE_THRESHOLDS,
        VOL_THRESHOLDS,
        TAKE_PROFITS,
        STOP_LOSSES,
    )

    for index, (price_th, vol_th, tp, sl) in enumerate(combinations, start=1):
        run_folder = Path(TUNING_FOLDER) / (
            f"run_{index}_price_{price_th}_vol_{vol_th}_tp_{tp}_sl_{abs(sl)}"
        )

        run_folder.mkdir(parents=True, exist_ok=True)

        print("\n================ STRATEGY TEST ================")
        print(f"Run folder: {run_folder}")
        print(f"price={price_th}, vol={vol_th}, TP={tp}, SL={sl}")
        print("==============================================")

        cfg = LiveSchedulerConfig(
            sleep_seconds=0,
            max_cycles=1,
            initial_cash=INITIAL_CASH,
            price_threshold=price_th,
            volatility_threshold=vol_th,
            take_profit_pct=tp,
            stop_loss_pct=sl,
            run_visual_report=False,
            show_plots=False,
            output_folder=str(run_folder),
            option_chain_file=str(run_folder / "live_eth_option_chain.csv"),
            candidates_file=str(run_folder / "live_backtest_candidates.csv"),
            paper_positions_file=str(run_folder / "paper_open_positions.csv"),
            paper_trade_history_file=str(run_folder / "paper_trade_history.csv"),
            paper_cash_file=str(run_folder / "paper_cash.csv"),
        )

        try:
            run_scanner_cycle(cfg)

            paper_cfg = build_paper_config(cfg)

            manage_paper_positions(
                trader_config=paper_cfg,
                take_profit_pct=tp,
                stop_loss_pct=sl,
            )

            open_paper_trades(paper_cfg)

            summary = summarize_run(run_folder)

            summary.update(
                {
                    "run_folder": str(run_folder),
                    "price_threshold": price_th,
                    "volatility_threshold": vol_th,
                    "take_profit_pct": tp,
                    "stop_loss_pct": sl,
                    "error": "",
                }
            )

        except Exception as error:
            summary = {
                "run_folder": str(run_folder),
                "price_threshold": price_th,
                "volatility_threshold": vol_th,
                "take_profit_pct": tp,
                "stop_loss_pct": sl,
                "cash": INITIAL_CASH,
                "equity": INITIAL_CASH,
                "open_positions": 0,
                "open_current_value": 0.0,
                "open_unrealized_pnl": 0.0,
                "closed_trades": 0,
                "total_realized_pnl": 0.0,
                "total_candidates": 0,
                "cheap_candidates": 0,
                "total_return_pct": 0.0,
                "error": str(error),
            }

        results.append(summary)

    results_df = pd.DataFrame(results)

    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_FILE, index=False)

    print("\n========== STRATEGY TUNING COMPLETE ==========")
    print(f"Saved results to: {OUTPUT_FILE}")
    print("============================================")

    return results_df


if __name__ == "__main__":
    run_strategy_tuning()