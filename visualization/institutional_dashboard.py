from __future__ import annotations
from pathlib import Path
import html
import pandas as pd


def _load(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def _section(title: str, data: pd.DataFrame, n: int = 20) -> str:
    body = '<p>No data available.</p>' if data.empty else data.tail(n).to_html(index=False, escape=True)
    return '<h2>' + html.escape(title) + '</h2>' + body


def generate_dashboard(output_file: str = 'outputs/institutional_research_dashboard.html') -> str:
    sections = [
        _section('Institutional Cycle Summary', _load('outputs/institutional_cycle_summary.csv')),
        _section('Portfolio Diagnostics', _load('outputs/portfolio_diagnostics.csv')),
        _section('IV Surface Diagnostics', _load('outputs/iv_surface_diagnostics.csv')),
        _section('Surface Scored Candidates', _load('outputs/live_backtest_candidates_surface_scored.csv')),
        _section('Scenario Portfolio PnL', _load('outputs/scenario_portfolio_pnl.csv')),
        _section('Portfolio Greeks', _load('outputs/portfolio_greeks.csv')),
    ]
    head = '<html><head><title>Institutional ETH Options Dashboard</title><style>body{font-family:Arial;margin:24px} table{border-collapse:collapse;font-size:12px} td,th{border:1px solid #ddd;padding:4px} th{background:#f3f3f3}</style></head><body><h1>Institutional ETH Options Dashboard</h1>'
    doc = head + ''.join(sections) + '</body></html>'
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(output_file).write_text(doc, encoding='utf-8')
    print(f'Saved dashboard to: {output_file}')
    return output_file


if __name__ == '__main__':
    generate_dashboard()
