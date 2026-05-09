"""
execution/paper_risk_metrics.py

Calculates paper risk metrics from the paper equity curve.
"""

from __future__ import annotations

from pathlib import Path
import math

import pandas as pd

EQUITY_CURVE_FILE = "outputs/paper_equity_curve.csv"
RISK_METRICS_FILE = "outputs/paper_risk_metrics.csv"


def load_equity_curve(file_path: str = EQUITY_CURVE_FILE) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def calculate_drawdown(equity: pd.Series) -> tuple[pd.Series, float, float]:
    if equity.empty:
        return pd.Series(dtype=float), 0.0, 0.0
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return drawdown, float(drawdown.min()), float(drawdown.iloc[-1])


def calculate_returns(equity: pd.Series) -> pd.Series:
    if equity.empty or len(equity) < 2:
        return pd.Series(dtype=float)
    return equity.pct_change().replace([math.inf, -math.inf], pd.NA).dropna()


def calculate_sharpe_ratio(returns: pd.Series, periods_per_year: int | None = None) -> float:
    if returns.empty or returns.std(ddof=1) == 0:
        return 0.0
    raw = float(returns.mean() / returns.std(ddof=1))
    if periods_per_year is None:
        return raw
    return float(raw * math.sqrt(periods_per_year))


def calculate_sortino_ratio(returns: pd.Series, periods_per_year: int | None = None) -> float:
    if returns.empty:
        return 0.0
    downside = returns[returns < 0]
    if downside.empty or downside.std(ddof=1) == 0:
        return 0.0
    raw = float(returns.mean() / downside.std(ddof=1))
    if periods_per_year is None:
        return raw
    return float(raw * math.sqrt(periods_per_year))


def calculate_risk_metrics(
    equity_curve_file: str = EQUITY_CURVE_FILE,
    output_file: str = RISK_METRICS_FILE,
    periods_per_year: int | None = None,
) -> pd.DataFrame:
    equity_curve = load_equity_curve(equity_curve_file)
    if equity_curve.empty or "equity" not in equity_curve.columns:
        result = pd.DataFrame([{"timestamp": pd.Timestamp.utcnow().isoformat(), "status": "no_equity_data"}])
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_file, index=False)
        return result

    equity = pd.to_numeric(equity_curve["equity"], errors="coerce").dropna()
    returns = calculate_returns(equity)
    drawdown, max_drawdown, current_drawdown = calculate_drawdown(equity)

    result = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp.utcnow().isoformat(),
                "observations": int(len(equity)),
                "current_equity": float(equity.iloc[-1]) if not equity.empty else 0.0,
                "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) >= 2 and equity.iloc[0] > 0 else 0.0,
                "max_drawdown": float(max_drawdown),
                "current_drawdown": float(current_drawdown),
                "mean_period_return": float(returns.mean()) if not returns.empty else 0.0,
                "period_volatility": float(returns.std(ddof=1)) if len(returns) > 1 else 0.0,
                "sharpe_ratio": calculate_sharpe_ratio(returns, periods_per_year),
                "sortino_ratio": calculate_sortino_ratio(returns, periods_per_year),
                "best_period_return": float(returns.max()) if not returns.empty else 0.0,
                "worst_period_return": float(returns.min()) if not returns.empty else 0.0,
            }
        ]
    )

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_file, index=False)
    print("Paper risk metrics generated.")
    return result


if __name__ == "__main__":
    calculate_risk_metrics()
