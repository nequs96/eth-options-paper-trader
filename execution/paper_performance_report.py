from __future__ import annotations

from pathlib import Path
import pandas as pd

from execution.paper_trader import PaperTraderConfig
from execution.paper_account_reconciliation import load_csv_if_exists, load_cash, open_only, calculate_open_current_value, calculate_open_cost_basis, calculate_unrealized_pnl, calculate_realized_pnl
from execution.paper_risk_metrics import calculate_risk_metrics

SUMMARY_FILE = "outputs/paper_performance_summary.csv"
PASSED_CANDIDATES_FILE = "outputs/paper_available_candidates_passed.csv"


def summarize_candidates(config: PaperTraderConfig, output_file: str = PASSED_CANDIDATES_FILE) -> tuple[int, pd.DataFrame]:
    candidates = load_csv_if_exists(config.candidates_file)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    if candidates.empty:
        pd.DataFrame().to_csv(output_file, index=False)
        return 0, pd.DataFrame()
    data = candidates.copy()
    preferred = ["instrument_name", "option_type", "expiry", "strike", "days_to_expiry", "market_price_usd", "model_price_usd", "price_diff_pct", "volatility_spread", "classification", "mci", "edge_score", "regime_score", "vol_score", "liquidity_score", "greek_score", "bid_ask_spread_pct", "delta", "gamma", "vega", "theta", "decision_reason"]
    cols = [c for c in preferred if c in data.columns]
    export = data[cols].copy() if cols else data.copy()
    if "mci" in export.columns:
        export["mci"] = pd.to_numeric(export["mci"], errors="coerce")
        export = export.sort_values("mci", ascending=False)
    export.to_csv(output_file, index=False)
    return int(len(export)), export


def generate_paper_performance_report(config: PaperTraderConfig | None = None, output_file: str = SUMMARY_FILE) -> pd.DataFrame:
    if config is None:
        config = PaperTraderConfig()
    history = load_csv_if_exists(config.trade_history_file)
    positions_all = load_csv_if_exists(config.positions_file)
    positions = open_only(positions_all)
    cash = load_cash(config)
    open_value = calculate_open_current_value(positions)
    open_cost = calculate_open_cost_basis(positions)
    unrealized = calculate_unrealized_pnl(positions)
    realized = calculate_realized_pnl(history)
    equity = cash + open_value
    candidates_passed, candidate_table = summarize_candidates(config)
    risk_metrics = calculate_risk_metrics(config=config)
    risk_row = risk_metrics.iloc[-1].to_dict() if not risk_metrics.empty else {}
    row = {
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "cash": float(cash),
        "open_cost_basis": float(open_cost),
        "open_position_value": float(open_value),
        "estimated_equity": float(equity),
        "realized_pnl": float(realized),
        "unrealized_pnl": float(unrealized),
        "actual_pnl_vs_start": float(equity - config.initial_cash),
        "total_return": float(equity / config.initial_cash - 1.0) if config.initial_cash > 0 else 0.0,
        "open_positions": int(len(positions)),
        "available_candidates_passed": int(candidates_passed),
        "sharpe_ratio": float(risk_row.get("sharpe_ratio", 0.0) or 0.0),
        "sortino_ratio": float(risk_row.get("sortino_ratio", 0.0) or 0.0),
        "profit_factor": float(risk_row.get("profit_factor", 0.0) or 0.0),
        "profit_ratio": float(risk_row.get("profit_ratio", 0.0) or 0.0),
        "win_rate": float(risk_row.get("win_rate", 0.0) or 0.0),
        "max_drawdown": float(risk_row.get("max_drawdown", 0.0) or 0.0),
    }
    result = pd.DataFrame([row])
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_file, index=False)
    return result


def print_performance_report(report: pd.DataFrame) -> None:
    print("\n========== PAPER PERFORMANCE ==========")
    if report.empty:
        print("No performance data.")
        return
    row = report.iloc[-1]
    print(f"Estimated equity: ${float(row.get('estimated_equity', 0.0)):,.2f}")
    print(f"Actual PnL vs start: ${float(row.get('actual_pnl_vs_start', 0.0)):,.2f}")
    print(f"Total return: {float(row.get('total_return', 0.0)):.2%}")
    print(f"Realized PnL: ${float(row.get('realized_pnl', 0.0)):,.2f}")
    print(f"Unrealized PnL: ${float(row.get('unrealized_pnl', 0.0)):,.2f}")
    print(f"Sharpe: {float(row.get('sharpe_ratio', 0.0)):.4f}")
    print(f"Sortino: {float(row.get('sortino_ratio', 0.0)):.4f}")
    print(f"Profit factor: {float(row.get('profit_factor', 0.0)):.4f}")
    print(f"Profit ratio: {float(row.get('profit_ratio', 0.0)):.4f}")
    print(f"Available candidates passed: {int(row.get('available_candidates_passed', 0))}")
    print("=======================================")


if __name__ == "__main__":
    print_performance_report(generate_paper_performance_report())
