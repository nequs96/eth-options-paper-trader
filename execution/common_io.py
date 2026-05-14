from __future__ import annotations
from pathlib import Path
import pandas as pd


def ensure_outputs() -> None:
    Path('outputs').mkdir(parents=True, exist_ok=True)


def load_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors='coerce').fillna(default).astype(float)


def sf(value, default: float = 0.0) -> float:
    try:
        x = float(value)
    except Exception:
        return default
    return x if pd.notna(x) else default
