"""
visualization/backtest_plots.py

Visualization tools for ETH options backtest results.

This module reads:
- outputs/equity_curve.csv
- outputs/trade_log.csv

And creates:
- equity curve chart
- drawdown chart
- ETH spot price chart
- trade PnL distribution
- combined backtest report chart
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def load_equity_curve(file_path: str = "outputs/equity_curve.csv") -> pd.DataFrame:
    """
    Load equity curve CSV.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Equity curve file not found: {file_path}")

    data = pd.read_csv(path)

    if data.empty:
        raise ValueError("Equity curve file is empty.")

    required_columns = {"timestamp", "equity", "cash", "spot_price"}

    missing_columns = required_columns.difference(set(data.columns))

    if missing_columns:
        raise ValueError(f"Equity curve missing columns: {missing_columns}")

    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["equity"] = pd.to_numeric(data["equity"], errors="coerce")
    data["cash"] = pd.to_numeric(data["cash"], errors="coerce")
    data["spot_price"] = pd.to_numeric(data["spot_price"], errors="coerce")

    data = data.dropna(subset=["timestamp", "equity", "cash", "spot_price"])

    if data.empty:
        raise ValueError("Equity curve became empty after cleaning.")

    return data


def load_trade_log(file_path: str = "outputs/trade_log.csv") -> pd.DataFrame:
    """
    Load trade log CSV.

    If no trades exist, this returns an empty DataFrame.
    """

    path = Path(file_path)

    if not path.exists():
        return pd.DataFrame()

    data = pd.read_csv(path)

    if data.empty:
        return data

    if "pnl" in data.columns:
        data["pnl"] = pd.to_numeric(data["pnl"], errors="coerce")

    if "entry_date" in data.columns:
        data["entry_date"] = pd.to_datetime(data["entry_date"], errors="coerce")

    if "exit_date" in data.columns:
        data["exit_date"] = pd.to_datetime(data["exit_date"], errors="coerce")

    return data


def calculate_drawdown(equity: pd.Series) -> pd.Series:
    """
    Calculate drawdown series.
    """

    if equity.empty:
        raise ValueError("equity cannot be empty.")

    rolling_peak = equity.cummax()
    drawdown = equity / rolling_peak - 1.0

    return drawdown


def plot_equity_curve(
    equity_data: pd.DataFrame,
    show_plot: bool = True,
    save_path: str | None = "outputs/equity_curve.png",
) -> None:
    """
    Plot portfolio equity over time.
    """

    plt.figure(figsize=(12, 6))
    plt.plot(equity_data["timestamp"], equity_data["equity"], label="Portfolio equity")
    plt.title("Backtest Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend()
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_drawdown(
    equity_data: pd.DataFrame,
    show_plot: bool = True,
    save_path: str | None = "outputs/drawdown.png",
) -> None:
    """
    Plot portfolio drawdown over time.
    """

    drawdown = calculate_drawdown(equity_data["equity"])

    plt.figure(figsize=(12, 5))
    plt.plot(equity_data["timestamp"], drawdown, label="Drawdown")
    plt.title("Backtest Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_spot_price(
    equity_data: pd.DataFrame,
    show_plot: bool = True,
    save_path: str | None = "outputs/eth_spot_price.png",
) -> None:
    """
    Plot ETH spot price over backtest period.
    """

    plt.figure(figsize=(12, 6))
    plt.plot(equity_data["timestamp"], equity_data["spot_price"], label="ETH spot price")
    plt.title("ETH Spot Price During Backtest")
    plt.xlabel("Date")
    plt.ylabel("ETH Price")
    plt.legend()
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_trade_pnl_distribution(
    trade_log: pd.DataFrame,
    show_plot: bool = True,
    save_path: str | None = "outputs/trade_pnl_distribution.png",
) -> None:
    """
    Plot trade PnL histogram.
    """

    if trade_log.empty or "pnl" not in trade_log.columns:
        print("No trade PnL data available. Skipping PnL distribution plot.")
        return

    pnl = trade_log["pnl"].dropna()

    if pnl.empty:
        print("Trade PnL column is empty. Skipping PnL distribution plot.")
        return

    plt.figure(figsize=(10, 6))
    plt.hist(pnl, bins=30)
    plt.title("Trade PnL Distribution")
    plt.xlabel("PnL")
    plt.ylabel("Frequency")
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_backtest_report(
    equity_data: pd.DataFrame,
    trade_log: pd.DataFrame,
    show_plot: bool = True,
    save_path: str | None = "outputs/backtest_report.png",
) -> None:
    """
    Create combined backtest report chart.
    """

    drawdown = calculate_drawdown(equity_data["equity"])

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=False)

    axes[0].plot(
        equity_data["timestamp"],
        equity_data["equity"],
        label="Portfolio equity",
    )
    axes[0].set_title("Portfolio Equity")
    axes[0].set_ylabel("Equity")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(
        equity_data["timestamp"],
        drawdown,
        label="Drawdown",
    )
    axes[1].set_title("Portfolio Drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(
        equity_data["timestamp"],
        equity_data["spot_price"],
        label="ETH spot price",
    )
    axes[2].set_title("ETH Spot Price")
    axes[2].set_xlabel("Date")
    axes[2].set_ylabel("ETH Price")
    axes[2].legend()
    axes[2].grid(True)

    if not trade_log.empty and "exit_date" in trade_log.columns and "pnl" in trade_log.columns:
        clean_trades = trade_log.dropna(subset=["exit_date", "pnl"])

        if not clean_trades.empty:
            winning_trades = clean_trades[clean_trades["pnl"] > 0]
            losing_trades = clean_trades[clean_trades["pnl"] < 0]

            axes[0].scatter(
                winning_trades["exit_date"],
                [equity_data["equity"].iloc[-1]] * len(winning_trades),
                marker="^",
                label="Winning trade exits",
            )

            axes[0].scatter(
                losing_trades["exit_date"],
                [equity_data["equity"].iloc[-1]] * len(losing_trades),
                marker="v",
                label="Losing trade exits",
            )

            axes[0].legend()

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close()


def generate_all_backtest_plots(
    equity_file: str = "outputs/equity_curve.csv",
    trade_file: str = "outputs/trade_log.csv",
    show_plot: bool = True,
) -> None:
    """
    Generate all backtest plots from output CSV files.
    """

    equity_data = load_equity_curve(equity_file)
    trade_log = load_trade_log(trade_file)

    plot_equity_curve(equity_data, show_plot=show_plot)
    plot_drawdown(equity_data, show_plot=show_plot)
    plot_spot_price(equity_data, show_plot=show_plot)
    plot_trade_pnl_distribution(trade_log, show_plot=show_plot)
    plot_backtest_report(equity_data, trade_log, show_plot=show_plot)

    print("========== BACKTEST PLOTS GENERATED ==========")
    print("Saved charts:")
    print("outputs/equity_curve.png")
    print("outputs/drawdown.png")
    print("outputs/eth_spot_price.png")
    print("outputs/trade_pnl_distribution.png")
    print("outputs/backtest_report.png")
    print("==============================================")


if __name__ == "__main__":
    generate_all_backtest_plots(
        equity_file="outputs/equity_curve.csv",
        trade_file="outputs/trade_log.csv",
        show_plot=True,
    )