"""Paper risk metrics from equity curve and closed trade history."""
from __future__ import annotations

from pathlib import Path
import math
import pandas as pd

from execution.paper_trader import PaperTraderConfig
from execution.paper_account_reconciliation import load_csv_if_exists, numeric_column, closed_only

EQUITY_CURVE_FILE = "outputs/paper_equity_curve.csv"
RISK_METRICS_FILE = "outputs/paper_risk_metrics.csv"


def safe_ratio(num: float, den: float) -> float:
    return 0.0 if den == 0 or not math.isfinite(den) else float(num / den)


def infer_periods_per_year(df: pd.DataFrame, fallback: int = 365) -> int:
    if "timestamp" not in df.columns or len(df) < 3:
        return fallback
    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dropna().sort_values()
    seconds = ts.diff().dt.total_seconds().dropna()
    seconds = seconds[seconds > 0]
    if seconds.empty:
        return fallback
    return max(1, int(round((365 * 24 * 3600) / float(seconds.median()))))


def trade_statistics(config: PaperTraderConfig) -> dict[str, float | int]:
    history = closed_only(load_csv_if_exists(config.trade_history_file))
    if history.empty:
        return {"closed_trades": 0, "winning_trades": 0, "losing_trades": 0, "win_rate": 0.0, "gross_profit": 0.0, "gross_loss": 0.0, "net_realized_pnl": 0.0, "profit_factor": 0.0, "profit_ratio": 0.0, "average_win": 0.0, "average_loss": 0.0}
    pnl = numeric_column(history, ["pnl_usd", "pnl"], 0.0)
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    gross_profit, gross_loss = float(wins.sum()), float(losses.sum())
    avg_win = float(wins.mean()) if not wins.empty else 0.0
    avg_loss = float(losses.mean()) if not losses.empty else 0.0
    return {
        "closed_trades": int(len(pnl)),
        "winning_trades": int(len(wins)),
        "losing_trades": int(len(losses)),
        "win_rate": safe_ratio(float(len(wins)), float(len(pnl))),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "net_realized_pnl": float(pnl.sum()),
        "profit_factor": safe_ratio(gross_profit, abs(gross_loss)),
        "profit_ratio": safe_ratio(avg_win, abs(avg_loss)),
        "average_win": avg_win,
        "average_loss": avg_loss,
    }


def calculate_risk_metrics(equity_curve_file: str = EQUITY_CURVE_FILE, output_file: str = RISK_METRICS_FILE, periods_per_year: int | None = None, config: PaperTraderConfig | None = None) -> pd.DataFrame:
    if config is None:
        config = PaperTraderConfig()
    df = load_csv_if_exists(equity_curve_file)
    if df.empty or "equity" not in df.columns:
        result = pd.DataFrame([{"timestamp": pd.Timestamp.utcnow().isoformat(), "status": "no_equity_data"}])
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_file, index=False)
        return result
    equity = pd.to_numeric(df["equity"], errors="coerce").dropna()
    if equity.empty:
        result = pd.DataFrame([{"timestamp": pd.Timestamp.utcnow().isoformat(), "status": "no_valid_equity"}])
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_file, index=False)
        return result
    if periods_per_year is None:
        periods_per_year = infer_periods_per_year(df)
    returns = equity.pct_change().replace([math.inf, -math.inf], pd.NA).dropna()
    downside = returns[returns < 0]
    drawdown = equity / equity.cummax() - 1.0
    ret_std = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    down_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    mean_ret = float(returns.mean()) if not returns.empty else 0.0
    sharpe = safe_ratio(mean_ret, ret_std) * math.sqrt(periods_per_year) if ret_std > 0 else 0.0
    sortino = safe_ratio(mean_ret, down_std) * math.sqrt(periods_per_year) if down_std > 0 else 0.0
    start, current = float(equity.iloc[0]), float(equity.iloc[-1])
    result = pd.DataFrame([{ "timestamp": pd.Timestamp.utcnow().isoformat(), "status": "ok", "observations": int(len(equity)), "periods_per_year": int(periods_per_year), "current_equity": current, "total_return": safe_ratio(current, start) - 1.0 if start > 0 else 0.0, "max_drawdown": float(drawdown.min()) if not drawdown.empty else 0.0, "current_drawdown": float(drawdown.iloc[-1]) if not drawdown.empty else 0.0, "mean_period_return": mean_ret, "volatility_period": ret_std, "sharpe_ratio": float(sharpe), "sortino_ratio": float(sortino), **trade_statistics(config)}])
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_file, index=False)
    return result


def print_risk_metrics_report(metrics: pd.DataFrame) -> None:
    print("\n========== PAPER RISK METRICS ==========")
    if metrics.empty or metrics.iloc[-1].get("status") != "ok":
        print("Not enough equity data yet.")
        return
    row = metrics.iloc[-1]
    print(f"Sharpe ratio: {float(row.get('sharpe_ratio', 0.0)):.4f}")
    print(f"Sortino ratio: {float(row.get('sortino_ratio', 0.0)):.4f}")
    print(f"Profit factor: {float(row.get('profit_factor', 0.0)):.4f}")
    print(f"Profit ratio: {float(row.get('profit_ratio', 0.0)):.4f}")
    print(f"Max drawdown: {float(row.get('max_drawdown', 0.0)):.2%}")
    print("========================================")


if __name__ == "__main__":
    print_risk_metrics_report(calculate_risk_metrics())
