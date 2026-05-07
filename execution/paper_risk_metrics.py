"""
execution/paper_risk_metrics.py

Calculates risk metrics for paper trading:

- equity curve
- returns
- max drawdown
- current drawdown
- Sharpe ratio
- Sortino ratio
- volatility
- best/worst equity point

Reads:
- outputs/paper_equity_curve.csv

Creates:
- outputs/paper_risk_metrics.csv

Important:
This is for paper trading analysis only.
"""

from pathlib import Path
import math

import pandas as pd


EQUITY_CURVE_FILE = "outputs/paper_equity_curve.csv"
RISK_METRICS_FILE = "outputs/paper_risk_metrics.csv"

# For paper-trading intraday/simple cycle metrics.
# If your scheduler runs every 15 min, annualization is approximate.
PERIODS_PER_YEAR = 365


def load_equity_curve(file_path: str = EQUITY_CURVE_FILE) -> pd.DataFrame:
    """
    Load paper equity curve.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Equity curve file not found: {file_path}. "
            "Run: python -m execution.paper_equity_curve"
        )

    if path.stat().st_size == 0:
        raise ValueError("Equity curve file is empty.")

    data = pd.read_csv(path)

    if data.empty:
        raise ValueError("Equity curve is empty.")

    required_columns = {"timestamp", "equity"}
    missing = required_columns - set(data.columns)

    if missing:
        raise ValueError(f"Equity curve missing columns: {missing}")

    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data["equity"] = pd.to_numeric(data["equity"], errors="coerce")

    data = data.dropna(subset=["timestamp", "equity"])
    data = data.sort_values("timestamp").reset_index(drop=True)

    if data.empty:
        raise ValueError("No valid equity curve rows after cleaning.")

    return data


def calculate_drawdown(equity: pd.Series) -> tuple[pd.Series, float, float]:
    """
    Calculate drawdown series, max drawdown, and current drawdown.
    """

    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0

    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0
    current_drawdown = float(drawdown.iloc[-1]) if not drawdown.empty else 0.0

    return drawdown, max_drawdown, current_drawdown


def calculate_returns(equity: pd.Series) -> pd.Series:
    """
    Calculate percentage returns from equity curve.
    """

    returns = equity.pct_change()
    returns = returns.replace([math.inf, -math.inf], pd.NA)
    returns = returns.dropna()

    return returns


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> float:
    """
    Calculate annualized Sharpe ratio.

    Formula:
        Sharpe = mean(excess returns) / std(returns) * sqrt(periods_per_year)
    """

    if returns.empty:
        return 0.0

    period_risk_free = risk_free_rate / periods_per_year
    excess_returns = returns - period_risk_free

    std = excess_returns.std(ddof=1)

    if std == 0 or pd.isna(std):
        return 0.0

    return float(excess_returns.mean() / std * math.sqrt(periods_per_year))


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> float:
    """
    Calculate annualized Sortino ratio.

    Sortino only penalizes downside volatility.
    """

    if returns.empty:
        return 0.0

    period_risk_free = risk_free_rate / periods_per_year
    excess_returns = returns - period_risk_free

    downside = excess_returns[excess_returns < 0]

    if downside.empty:
        return math.inf if excess_returns.mean() > 0 else 0.0

    downside_std = downside.std(ddof=1)

    if downside_std == 0 or pd.isna(downside_std):
        return 0.0

    return float(excess_returns.mean() / downside_std * math.sqrt(periods_per_year))


def calculate_risk_metrics(
    equity_curve_file: str = EQUITY_CURVE_FILE,
    output_file: str = RISK_METRICS_FILE,
    risk_free_rate: float = 0.0,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> pd.DataFrame:
    """
    Calculate and save risk metrics.
    """

    equity_df = load_equity_curve(equity_curve_file)

    equity = equity_df["equity"]

    returns = calculate_returns(equity)
    drawdown, max_drawdown, current_drawdown = calculate_drawdown(equity)

    equity_df["drawdown"] = drawdown
    equity_df["return"] = equity.pct_change().fillna(0.0)

    starting_equity = float(equity.iloc[0])
    ending_equity = float(equity.iloc[-1])

    total_return = ending_equity / starting_equity - 1.0 if starting_equity > 0 else 0.0

    volatility = (
        float(returns.std(ddof=1) * math.sqrt(periods_per_year))
        if not returns.empty and returns.std(ddof=1) != 0
        else 0.0
    )

    sharpe = calculate_sharpe_ratio(
        returns=returns,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
    )

    sortino = calculate_sortino_ratio(
        returns=returns,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
    )

    best_equity = float(equity.max())
    worst_equity = float(equity.min())

    best_return = float(returns.max()) if not returns.empty else 0.0
    worst_return = float(returns.min()) if not returns.empty else 0.0

    metrics = {
        "starting_equity": starting_equity,
        "ending_equity": ending_equity,
        "total_return": total_return,
        "annualized_volatility": volatility,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_drawdown,
        "current_drawdown": current_drawdown,
        "best_equity": best_equity,
        "worst_equity": worst_equity,
        "best_period_return": best_return,
        "worst_period_return": worst_return,
        "equity_points": int(len(equity_df)),
    }

    metrics_df = pd.DataFrame([metrics])

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_file, index=False)

    # Also overwrite equity curve with drawdown/return columns added.
    equity_df.to_csv(equity_curve_file, index=False)

    print("========== PAPER RISK METRICS ==========")
    print(f"Starting equity:        ${metrics['starting_equity']:,.2f}")
    print(f"Ending equity:          ${metrics['ending_equity']:,.2f}")
    print(f"Total return:           {metrics['total_return']:.2%}")
    print("----------------------------------------")
    print(f"Sharpe ratio:           {metrics['sharpe_ratio']:.4f}")
    print(f"Sortino ratio:          {metrics['sortino_ratio']:.4f}")
    print(f"Annualized volatility:  {metrics['annualized_volatility']:.2%}")
    print("----------------------------------------")
    print(f"Max drawdown:           {metrics['max_drawdown']:.2%}")
    print(f"Current drawdown:       {metrics['current_drawdown']:.2%}")
    print(f"Best period return:     {metrics['best_period_return']:.2%}")
    print(f"Worst period return:    {metrics['worst_period_return']:.2%}")
    print("----------------------------------------")
    print(f"Saved metrics to:       {output_file}")
    print(f"Updated equity curve:   {equity_curve_file}")
    print("========================================")

    return metrics_df


if __name__ == "__main__":
    calculate_risk_metrics()