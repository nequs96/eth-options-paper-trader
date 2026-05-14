from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd


@dataclass(frozen=True)
class DeriskConfig:
    positions_file: str = "outputs/paper_open_positions.csv"
    breaches_file: str = "outputs/risk_limit_breaches.csv"
    output_file: str = "outputs/derisking_recommendations.csv"
    preview_file: str = "outputs/portfolio_after_derisk_preview.csv"
    max_reductions: int = 4


def _load_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except Exception as exc:
        print(f"Warning: could not read {path}: {exc}")
        return pd.DataFrame()


def _numeric_col(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float)


def _first_numeric_col(df: pd.DataFrame, cols: list[str], default: float = 0.5) -> pd.Series:
    """
    Returns the first available numeric column from cols.
    This avoids the old bug where a Series was passed as the default to another Series.
    """
    for col in cols:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            if s.notna().any():
                return s.fillna(default).astype(float)
    return pd.Series([default] * len(df), index=df.index, dtype=float)


def build_derisking_recommendations(config: DeriskConfig | None = None) -> pd.DataFrame:
    cfg = config or DeriskConfig()
    Path("outputs").mkdir(exist_ok=True)

    positions = _load_csv(cfg.positions_file)
    breaches = _load_csv(cfg.breaches_file)

    if positions.empty:
        out = pd.DataFrame(columns=["instrument_name", "recommended_action", "derisk_score", "derisk_reason"])
        out.to_csv(cfg.output_file, index=False)
        out.to_csv(cfg.preview_file, index=False)
        print("Derisking recommendations complete. rows=0 reason=no_positions")
        return out

    if "status" in positions.columns:
        open_positions = positions[positions["status"].astype(str).str.lower().eq("open")].copy()
    else:
        open_positions = positions.copy()
    if open_positions.empty:
        out = pd.DataFrame(columns=["instrument_name", "recommended_action", "derisk_score", "derisk_reason"])
        out.to_csv(cfg.output_file, index=False)
        open_positions.to_csv(cfg.preview_file, index=False)
        print("Derisking recommendations complete. rows=0 reason=no_open_positions")
        return out

    pnl = _numeric_col(open_positions, "unrealized_pnl_pct", 0.0)
    theta = _numeric_col(open_positions, "theta", 0.0)
    dte = _numeric_col(open_positions, "days_to_expiry", 999.0).clip(lower=0.1)

    score = _first_numeric_col(
        open_positions,
        ["institutional_edge_score", "mci", "confidence_score", "portfolio_candidate_score"],
        default=0.5,
    ).clip(lower=0.0, upper=1.0)

    # Higher score = more urgent to reduce.
    # Components:
    # - losing positions rank higher
    # - negative theta rank higher
    # - near-expiry positions rank higher
    # - low edge/confidence rank higher
    open_positions["derisk_score"] = (
        (-pnl).clip(lower=-1.0, upper=1.0) * 0.35
        + (-theta).clip(lower=0.0) * 0.05
        + (1.0 / dte) * 0.20
        + (1.0 - score).clip(lower=0.0, upper=1.0) * 0.30
    )

    open_positions["derisk_reason"] = "portfolio_risk_breach" if not breaches.empty else "risk_reduction_candidate"
    open_positions["recommended_action"] = "REVIEW_CLOSE_OR_REDUCE"

    out = open_positions.sort_values("derisk_score", ascending=False).head(cfg.max_reductions).copy()

    if "instrument_name" in open_positions.columns and "instrument_name" in out.columns:
        preview = open_positions[
            ~open_positions["instrument_name"].astype(str).isin(out["instrument_name"].astype(str))
        ].copy()
    else:
        preview = open_positions.iloc[cfg.max_reductions:].copy()

    out.to_csv(cfg.output_file, index=False)
    preview.to_csv(cfg.preview_file, index=False)

    print(f"Derisking recommendations complete. rows={len(out)}")
    return out


if __name__ == "__main__":
    build_derisking_recommendations()
