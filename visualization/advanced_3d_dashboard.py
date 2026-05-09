"""
visualization/advanced_3d_dashboard.py

Advanced interactive Plotly visualizations for the ETH options paper-trading system.

Creates:
- 3D implied volatility surface/scatter
- 3D candidate opportunity map
- 3D portfolio exposure map
- 3D risk allocation map
- combined HTML dashboard

This module is visualization-only. It does not place trades.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

OUTPUT_FOLDER = "outputs"
OPTION_CHAIN_FILE = "outputs/live_eth_option_chain.csv"
RAW_CANDIDATES_FILE = "outputs/live_backtest_candidates.csv"
FILTERED_CANDIDATES_FILE = "outputs/live_backtest_candidates_filtered.csv"
POSITIONS_FILE = "outputs/paper_open_positions.csv"
EQUITY_CURVE_FILE = "outputs/paper_equity_curve.csv"
DASHBOARD_FILE = "outputs/advanced_options_dashboard.html"


def ensure_output_folder(folder: str = OUTPUT_FOLDER) -> None:
    Path(folder).mkdir(parents=True, exist_ok=True)


def load_csv_if_exists(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists() or file_path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def to_numeric_columns(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = data.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def first_existing_column(data: pd.DataFrame, columns: list[str]) -> str | None:
    for column in columns:
        if column in data.columns:
            return column
    return None


def normalize_option_type(value: Any) -> str:
    text = str(value).lower().strip()
    if text in {"c", "call", "calls"}:
        return "call"
    if text in {"p", "put", "puts"}:
        return "put"
    return text


def add_option_type_if_missing(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    if "option_type" in result.columns:
        result["option_type"] = result["option_type"].apply(normalize_option_type)
        return result
    if "instrument_name" in result.columns:
        result["option_type"] = result["instrument_name"].astype(str).str.split("-").str[-1].apply(normalize_option_type)
    return result


def add_expiry_if_missing(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    if "expiry" not in result.columns and "instrument_name" in result.columns:
        result["expiry"] = result["instrument_name"].astype(str).str.split("-").str[1]
    return result


def prepare_option_chain(option_chain: pd.DataFrame) -> pd.DataFrame:
    data = option_chain.copy()
    data = add_option_type_if_missing(data)
    data = add_expiry_if_missing(data)
    data = to_numeric_columns(
        data,
        [
            "strike",
            "strike_price",
            "days_to_expiry",
            "mark_iv",
            "iv",
            "implied_volatility",
            "underlying_price_usd",
            "spot_price",
            "mark_price_usd",
            "bid_price_usd",
            "ask_price_usd",
            "delta",
            "gamma",
            "vega",
            "theta",
        ],
    )

    strike_col = first_existing_column(data, ["strike", "strike_price"])
    iv_col = first_existing_column(data, ["mark_iv", "iv", "implied_volatility"])

    if strike_col is not None and strike_col != "strike":
        data["strike"] = data[strike_col]
    if iv_col is not None and iv_col != "implied_volatility":
        data["implied_volatility"] = data[iv_col]

    if "implied_volatility" in data.columns:
        # Deribit IV may arrive as 65 for 65%, or 0.65 for 65%.
        data["implied_volatility"] = data["implied_volatility"].apply(
            lambda x: x / 100.0 if pd.notna(x) and x > 5 else x
        )

    return data


def prepare_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    data = candidates.copy()
    data = add_option_type_if_missing(data)
    data = add_expiry_if_missing(data)
    data = to_numeric_columns(
        data,
        [
            "strike",
            "strike_price",
            "days_to_expiry",
            "market_price_usd",
            "model_price_usd",
            "theoretical_price",
            "price_diff_pct",
            "volatility_spread",
            "combined_score",
            "mispricing_score",
            "portfolio_candidate_score",
            "confidence_score",
            "dynamic_risk_pct",
            "bid_ask_spread_pct",
            "delta",
            "gamma",
            "vega",
            "theta",
        ],
    )

    strike_col = first_existing_column(data, ["strike", "strike_price"])
    if strike_col is not None and strike_col != "strike":
        data["strike"] = data[strike_col]

    score_col = first_existing_column(data, ["portfolio_candidate_score", "combined_score", "mispricing_score"])
    if score_col is not None:
        data["visual_score"] = pd.to_numeric(data[score_col], errors="coerce").abs()
    else:
        data["visual_score"] = 0.0

    if "classification" in data.columns:
        data["classification"] = data["classification"].astype(str).str.lower().str.strip()

    return data


def prepare_positions(positions: pd.DataFrame) -> pd.DataFrame:
    data = positions.copy()
    data = add_option_type_if_missing(data)
    data = add_expiry_if_missing(data)
    data = to_numeric_columns(
        data,
        [
            "strike",
            "strike_price",
            "days_to_expiry",
            "entry_price_usd",
            "current_price_usd",
            "capital_at_risk",
            "quantity",
            "unrealized_pnl_usd",
            "unrealized_pnl_pct",
            "portfolio_candidate_score",
            "confidence_score",
            "dynamic_risk_pct",
            "highest_profit_pct",
            "trailing_stop_price_usd",
            "delta",
            "gamma",
            "vega",
            "theta",
        ],
    )
    strike_col = first_existing_column(data, ["strike", "strike_price"])
    if strike_col is not None and strike_col != "strike":
        data["strike"] = data[strike_col]
    if "status" in data.columns:
        data = data[data["status"].astype(str).str.lower().eq("open")]
    return data.reset_index(drop=True)


def color_by_option_type(option_types: pd.Series) -> list[str]:
    colors = []
    for value in option_types.astype(str).str.lower():
        if value == "call":
            colors.append("#00CC96")
        elif value == "put":
            colors.append("#EF553B")
        else:
            colors.append("#636EFA")
    return colors


def create_empty_figure(title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text="No data available",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 20},
    )
    fig.update_layout(title=title)
    return fig


def create_volatility_surface_figure(option_chain: pd.DataFrame, option_type: str = "call") -> go.Figure:
    data = prepare_option_chain(option_chain)
    required = {"strike", "days_to_expiry", "implied_volatility", "option_type"}
    if data.empty or not required.issubset(data.columns):
        return create_empty_figure(f"{option_type.upper()} IV Surface")

    filtered = data[data["option_type"].astype(str).str.lower().eq(option_type.lower())].copy()
    filtered = filtered.dropna(subset=["strike", "days_to_expiry", "implied_volatility"])
    if filtered.empty:
        return create_empty_figure(f"{option_type.upper()} IV Surface")

    pivot = filtered.pivot_table(
        index="days_to_expiry",
        columns="strike",
        values="implied_volatility",
        aggfunc="mean",
    ).sort_index().sort_index(axis=1)

    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=pivot.columns.astype(float),
            y=pivot.index.astype(float),
            z=pivot.values * 100.0,
            colorscale="Viridis",
            colorbar={"title": "IV %"},
            hovertemplate="Strike=%{x}<br>DTE=%{y:.2f}<br>IV=%{z:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{option_type.upper()} Implied Volatility Surface",
        scene={
            "xaxis_title": "Strike",
            "yaxis_title": "Days to Expiry",
            "zaxis_title": "Implied Volatility (%)",
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 50},
    )
    return fig


def create_candidate_3d_map(candidates: pd.DataFrame, title: str = "3D Candidate Opportunity Map") -> go.Figure:
    data = prepare_candidates(candidates)
    required = {"strike", "days_to_expiry", "visual_score"}
    if data.empty or not required.issubset(data.columns):
        return create_empty_figure(title)

    data = data.dropna(subset=["strike", "days_to_expiry", "visual_score"])
    if data.empty:
        return create_empty_figure(title)

    marker_size = 6 + 22 * (data["visual_score"] / max(data["visual_score"].max(), 1e-9)).clip(0, 1)
    color_values = data["visual_score"]

    hover_cols = [
        "instrument_name",
        "option_type",
        "classification",
        "market_price_usd",
        "model_price_usd",
        "price_diff_pct",
        "volatility_spread",
        "bid_ask_spread_pct",
        "visual_score",
    ]
    hover_text = []
    for _, row in data.iterrows():
        lines = []
        for col in hover_cols:
            if col in data.columns:
                value = row.get(col)
                if pd.notna(value):
                    lines.append(f"{col}: {value}")
        hover_text.append("<br>".join(lines))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=data["strike"],
            y=data["days_to_expiry"],
            z=data["visual_score"],
            mode="markers",
            marker={
                "size": marker_size,
                "color": color_values,
                "colorscale": "Turbo",
                "opacity": 0.85,
                "colorbar": {"title": "Score"},
            },
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        scene={
            "xaxis_title": "Strike",
            "yaxis_title": "Days to Expiry",
            "zaxis_title": "Opportunity Score",
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 50},
    )
    return fig


def create_portfolio_3d_map(positions: pd.DataFrame) -> go.Figure:
    data = prepare_positions(positions)
    required = {"strike", "days_to_expiry", "capital_at_risk"}
    if data.empty or not required.issubset(data.columns):
        return create_empty_figure("3D Open Portfolio Exposure")

    data = data.dropna(subset=["strike", "days_to_expiry", "capital_at_risk"])
    if data.empty:
        return create_empty_figure("3D Open Portfolio Exposure")

    pnl_col = "unrealized_pnl_pct" if "unrealized_pnl_pct" in data.columns else None
    z = data[pnl_col] * 100.0 if pnl_col else pd.Series([0.0] * len(data), index=data.index)
    marker_size = 8 + 22 * (data["capital_at_risk"] / max(data["capital_at_risk"].max(), 1e-9)).clip(0, 1)

    hover_text = []
    for _, row in data.iterrows():
        hover_text.append(
            "<br>".join(
                [
                    f"instrument: {row.get('instrument_name', '')}",
                    f"type: {row.get('option_type', '')}",
                    f"strike: {row.get('strike', '')}",
                    f"DTE: {row.get('days_to_expiry', '')}",
                    f"risk: ${row.get('capital_at_risk', 0):.2f}",
                    f"PnL %: {row.get('unrealized_pnl_pct', 0):.2%}",
                    f"dynamic risk: {row.get('dynamic_risk_pct', 0):.2%}",
                    f"confidence: {row.get('confidence_score', 0):.2f}",
                ]
            )
        )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=data["strike"],
            y=data["days_to_expiry"],
            z=z,
            mode="markers",
            marker={
                "size": marker_size,
                "color": z,
                "colorscale": "RdYlGn",
                "opacity": 0.9,
                "colorbar": {"title": "PnL %"},
            },
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
        )
    )
    fig.update_layout(
        title="3D Open Portfolio Exposure",
        scene={
            "xaxis_title": "Strike",
            "yaxis_title": "Days to Expiry",
            "zaxis_title": "Unrealized PnL (%)",
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 50},
    )
    return fig


def create_risk_allocation_3d(positions: pd.DataFrame) -> go.Figure:
    data = prepare_positions(positions)
    required = {"strike", "days_to_expiry", "dynamic_risk_pct"}
    if data.empty or not required.issubset(data.columns):
        return create_empty_figure("3D Dynamic Risk Allocation")

    data = data.dropna(subset=["strike", "days_to_expiry", "dynamic_risk_pct"])
    if data.empty:
        return create_empty_figure("3D Dynamic Risk Allocation")

    z = data["dynamic_risk_pct"] * 100.0
    confidence = data["confidence_score"] if "confidence_score" in data.columns else data["dynamic_risk_pct"]
    marker_size = 8 + 24 * (confidence / max(confidence.max(), 1e-9)).clip(0, 1)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=data["strike"],
            y=data["days_to_expiry"],
            z=z,
            mode="markers+text",
            text=data.get("option_type", pd.Series([""] * len(data))),
            textposition="top center",
            marker={
                "size": marker_size,
                "color": z,
                "colorscale": "Plasma",
                "opacity": 0.88,
                "colorbar": {"title": "Risk %"},
            },
            hovertemplate=(
                "Strike=%{x}<br>DTE=%{y:.2f}<br>Risk=%{z:.2f}%<br>"
                "Type=%{text}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="3D Dynamic Risk Allocation",
        scene={
            "xaxis_title": "Strike",
            "yaxis_title": "Days to Expiry",
            "zaxis_title": "Dynamic Risk per Position (%)",
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 50},
    )
    return fig


def create_equity_curve_figure(equity_curve: pd.DataFrame) -> go.Figure:
    if equity_curve.empty or "equity" not in equity_curve.columns:
        return create_empty_figure("Paper Equity Curve")

    data = equity_curve.copy()
    if "timestamp" in data.columns:
        x = pd.to_datetime(data["timestamp"], errors="coerce")
    else:
        x = list(range(len(data)))

    data["equity"] = pd.to_numeric(data["equity"], errors="coerce")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=data["equity"],
            mode="lines+markers",
            name="Equity",
            line={"width": 3},
        )
    )
    fig.update_layout(
        title="Paper Equity Curve",
        xaxis_title="Time",
        yaxis_title="Equity USD",
        margin={"l": 40, "r": 20, "b": 40, "t": 50},
    )
    return fig


def save_individual_figures(
    option_chain: pd.DataFrame,
    raw_candidates: pd.DataFrame,
    filtered_candidates: pd.DataFrame,
    positions: pd.DataFrame,
    equity_curve: pd.DataFrame,
    output_folder: str = OUTPUT_FOLDER,
) -> None:
    ensure_output_folder(output_folder)
    figures = {
        "advanced_call_iv_surface.html": create_volatility_surface_figure(option_chain, "call"),
        "advanced_put_iv_surface.html": create_volatility_surface_figure(option_chain, "put"),
        "advanced_raw_candidate_3d_map.html": create_candidate_3d_map(raw_candidates, "3D Raw Candidate Opportunity Map"),
        "advanced_filtered_candidate_3d_map.html": create_candidate_3d_map(filtered_candidates, "3D Filtered Candidate Opportunity Map"),
        "advanced_portfolio_3d_exposure.html": create_portfolio_3d_map(positions),
        "advanced_risk_allocation_3d.html": create_risk_allocation_3d(positions),
        "advanced_equity_curve.html": create_equity_curve_figure(equity_curve),
    }
    for filename, fig in figures.items():
        fig.write_html(str(Path(output_folder) / filename), include_plotlyjs="cdn")


def create_combined_dashboard(
    option_chain: pd.DataFrame,
    raw_candidates: pd.DataFrame,
    filtered_candidates: pd.DataFrame,
    positions: pd.DataFrame,
    equity_curve: pd.DataFrame,
) -> go.Figure:
    call_surface = create_volatility_surface_figure(option_chain, "call")
    candidate_map = create_candidate_3d_map(filtered_candidates, "3D Filtered Candidate Opportunity Map")
    portfolio_map = create_portfolio_3d_map(positions)
    risk_map = create_risk_allocation_3d(positions)

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "surface"}, {"type": "scatter3d"}], [{"type": "scatter3d"}, {"type": "scatter3d"}]],
        subplot_titles=(
            "CALL IV Surface",
            "Filtered Candidates",
            "Open Portfolio Exposure",
            "Dynamic Risk Allocation",
        ),
    )

    for trace in call_surface.data:
        fig.add_trace(trace, row=1, col=1)
    for trace in candidate_map.data:
        fig.add_trace(trace, row=1, col=2)
    for trace in portfolio_map.data:
        fig.add_trace(trace, row=2, col=1)
    for trace in risk_map.data:
        fig.add_trace(trace, row=2, col=2)

    fig.update_layout(
        title="Advanced ETH Options Paper Trading Dashboard",
        height=1100,
        showlegend=False,
        margin={"l": 10, "r": 10, "b": 10, "t": 80},
    )
    return fig


def generate_advanced_visual_report(
    option_chain_file: str = OPTION_CHAIN_FILE,
    raw_candidates_file: str = RAW_CANDIDATES_FILE,
    filtered_candidates_file: str = FILTERED_CANDIDATES_FILE,
    positions_file: str = POSITIONS_FILE,
    equity_curve_file: str = EQUITY_CURVE_FILE,
    output_file: str = DASHBOARD_FILE,
    save_individual: bool = True,
    show_plot: bool = False,
) -> go.Figure:
    ensure_output_folder(Path(output_file).parent.as_posix())

    option_chain = load_csv_if_exists(option_chain_file)
    raw_candidates = load_csv_if_exists(raw_candidates_file)
    filtered_candidates = load_csv_if_exists(filtered_candidates_file)
    positions = load_csv_if_exists(positions_file)
    equity_curve = load_csv_if_exists(equity_curve_file)

    if save_individual:
        save_individual_figures(
            option_chain=option_chain,
            raw_candidates=raw_candidates,
            filtered_candidates=filtered_candidates,
            positions=positions,
            equity_curve=equity_curve,
            output_folder=Path(output_file).parent.as_posix(),
        )

    dashboard = create_combined_dashboard(
        option_chain=option_chain,
        raw_candidates=raw_candidates,
        filtered_candidates=filtered_candidates,
        positions=positions,
        equity_curve=equity_curve,
    )
    dashboard.write_html(output_file, include_plotlyjs="cdn")

    print(f"Saved advanced dashboard to: {output_file}")

    if show_plot:
        dashboard.show()

    return dashboard


if __name__ == "__main__":
    generate_advanced_visual_report(show_plot=False)
