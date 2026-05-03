"""
backtesting/metrics.py

Performance metrics for ETH options backtesting.

This module calculates:
- total return
- max drawdown
- win rate
- average win
- average loss
- profit factor
- Sharpe ratio
- trade statistics

This file does not place trades.
It only evaluates backtest results.
"""

import math
from dataclasses import dataclass

import pandas as pd

from backtesting.portfolio import Trade


@dataclass
class PerformanceMetrics:
    """
    Container for backtest performance metrics.
    """

    initial_equity: float
    final_equity: float
    total_return: float
    max_drawdown: float
    number_of_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    average_win: float
    average_loss: float
    profit_factor: float
    sharpe_ratio: float


def calculate_total_return(
    initial_equity: float,
    final_equity: float,
) -> float:
    """
    Calculate total return.

    Formula:
        final_equity / initial_equity - 1
    """

    if initial_equity <= 0:
        raise ValueError("initial_equity must be greater than 0.")

    return float(final_equity / initial_equity - 1.0)


def calculate_drawdown_series(
    equity_curve: pd.Series,
) -> pd.Series:
    """
    Calculate drawdown series from an equity curve.

    Drawdown:
        equity / rolling_peak - 1
    """

    if equity_curve.empty:
        raise ValueError("equity_curve cannot be empty.")

    if (equity_curve <= 0).any():
        raise ValueError("equity_curve must contain positive values.")

    rolling_peak = equity_curve.cummax()
    drawdown = equity_curve / rolling_peak - 1.0

    return drawdown


def calculate_max_drawdown(
    equity_curve: pd.Series,
) -> float:
    """
    Calculate maximum drawdown.
    """

    drawdown = calculate_drawdown_series(equity_curve)

    return float(drawdown.min())


def calculate_trade_statistics(
    trades: list[Trade],
) -> dict[str, float | int]:
    """
    Calculate trade-level statistics.

    Returns:
    - number of trades
    - winning trades
    - losing trades
    - win rate
    - average win
    - average loss
    - profit factor
    """

    number_of_trades = len(trades)

    if number_of_trades == 0:
        return {
            "number_of_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "profit_factor": 0.0,
        }

    pnl_values = [trade.pnl for trade in trades]

    wins = [pnl for pnl in pnl_values if pnl > 0]
    losses = [pnl for pnl in pnl_values if pnl < 0]

    winning_trades = len(wins)
    losing_trades = len(losses)

    win_rate = winning_trades / number_of_trades

    average_win = sum(wins) / winning_trades if winning_trades > 0 else 0.0
    average_loss = sum(losses) / losing_trades if losing_trades > 0 else 0.0

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))

    if gross_loss == 0:
        profit_factor = math.inf if gross_profit > 0 else 0.0
    else:
        profit_factor = gross_profit / gross_loss

    return {
        "number_of_trades": number_of_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": float(win_rate),
        "average_win": float(average_win),
        "average_loss": float(average_loss),
        "profit_factor": float(profit_factor),
    }


def calculate_sharpe_ratio(
    equity_curve: pd.Series,
    periods_per_year: int = 365,
    risk_free_rate: float = 0.0,
) -> float:
    """
    Calculate annualized Sharpe ratio from an equity curve.

    Parameters
    ----------
    equity_curve : pd.Series
        Portfolio equity over time.
    periods_per_year : int
        Number of observations per year.
        For daily crypto data use 365.
        For hourly crypto data use 365 * 24.
    risk_free_rate : float
        Annual risk-free rate as decimal.

    Returns
    -------
    float
        Annualized Sharpe ratio.
    """

    if equity_curve.empty:
        raise ValueError("equity_curve cannot be empty.")

    returns = equity_curve.pct_change().dropna()

    if returns.empty:
        return 0.0

    period_risk_free_rate = risk_free_rate / periods_per_year
    excess_returns = returns - period_risk_free_rate

    volatility = excess_returns.std(ddof=1)

    if volatility == 0:
        return 0.0

    sharpe = excess_returns.mean() / volatility * math.sqrt(periods_per_year)

    return float(sharpe)


def calculate_performance_metrics(
    equity_curve: pd.Series,
    trades: list[Trade],
    periods_per_year: int = 365,
    risk_free_rate: float = 0.0,
) -> PerformanceMetrics:
    """
    Calculate full performance metrics.

    Parameters
    ----------
    equity_curve : pd.Series
        Portfolio equity over time.
    trades : list[Trade]
        Closed trades.
    periods_per_year : int
        Number of observations per year.
    risk_free_rate : float
        Annual risk-free rate.

    Returns
    -------
    PerformanceMetrics
        Full performance metrics object.
    """

    if equity_curve.empty:
        raise ValueError("equity_curve cannot be empty.")

    initial_equity = float(equity_curve.iloc[0])
    final_equity = float(equity_curve.iloc[-1])

    total_return = calculate_total_return(
        initial_equity=initial_equity,
        final_equity=final_equity,
    )

    max_drawdown = calculate_max_drawdown(equity_curve)

    trade_stats = calculate_trade_statistics(trades)

    sharpe_ratio = calculate_sharpe_ratio(
        equity_curve=equity_curve,
        periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate,
    )

    return PerformanceMetrics(
        initial_equity=initial_equity,
        final_equity=final_equity,
        total_return=float(total_return),
        max_drawdown=float(max_drawdown),
        number_of_trades=int(trade_stats["number_of_trades"]),
        winning_trades=int(trade_stats["winning_trades"]),
        losing_trades=int(trade_stats["losing_trades"]),
        win_rate=float(trade_stats["win_rate"]),
        average_win=float(trade_stats["average_win"]),
        average_loss=float(trade_stats["average_loss"]),
        profit_factor=float(trade_stats["profit_factor"]),
        sharpe_ratio=float(sharpe_ratio),
    )


def print_performance_metrics(
    metrics: PerformanceMetrics,
) -> None:
    """
    Print performance metrics in readable format.
    """

    print("========== BACKTEST PERFORMANCE ==========")
    print(f"Initial equity:      ${metrics.initial_equity:,.2f}")
    print(f"Final equity:        ${metrics.final_equity:,.2f}")
    print(f"Total return:        {metrics.total_return:.2%}")
    print(f"Max drawdown:        {metrics.max_drawdown:.2%}")
    print("------------------------------------------")
    print(f"Number of trades:    {metrics.number_of_trades}")
    print(f"Winning trades:      {metrics.winning_trades}")
    print(f"Losing trades:       {metrics.losing_trades}")
    print(f"Win rate:            {metrics.win_rate:.2%}")
    print(f"Average win:         ${metrics.average_win:,.2f}")
    print(f"Average loss:        ${metrics.average_loss:,.2f}")
    print(f"Profit factor:       {metrics.profit_factor:.4f}")
    print(f"Sharpe ratio:        {metrics.sharpe_ratio:.4f}")
    print("==========================================")


if __name__ == "__main__":
    # Standalone test

    from backtesting.portfolio import Portfolio

    portfolio = Portfolio(initial_cash=10_000.0)

    equity_values = [
        10_000.0,
        10_100.0,
        10_050.0,
        10_300.0,
        10_200.0,
        10_500.0,
    ]

    equity_curve = pd.Series(equity_values)

    portfolio.open_position(
        option_type="call",
        entry_price=200.0,
        quantity=0.5,
        strike_price=3200.0,
        days_to_expiry=30,
        direction="long",
    )

    portfolio.close_position(
        position_index=0,
        exit_price=260.0,
    )

    metrics = calculate_performance_metrics(
        equity_curve=equity_curve,
        trades=portfolio.closed_trades,
        periods_per_year=365,
        risk_free_rate=0.0,
    )

    print_performance_metrics(metrics)