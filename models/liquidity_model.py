"""
models/liquidity_model.py

Liquidity and execution quality model.

This model answers:
- Is the option realistically tradable?
- Is the spread likely to destroy edge?
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class LiquidityDecision:
    allowed: bool
    reason: str
    execution_score: float


def safe_float(x: Any) -> float | None:
    try:
        v = float(x)
        if v != v:
            return None
        return v
    except Exception:
        return None


def evaluate_liquidity(
    market_price_usd: Any,
    bid_ask_spread_pct: Any,
    open_interest: Any = None,
    min_price_usd: float = 10.0,
    max_spread_pct: float = 0.20,
) -> LiquidityDecision:
    price = safe_float(market_price_usd)
    spread = safe_float(bid_ask_spread_pct)

    if price is None or price < min_price_usd:
        return LiquidityDecision(False, "price_too_low", 0.0)

    if spread is None or spread > max_spread_pct:
        return LiquidityDecision(False, "spread_too_wide", 0.0)

    score = max(0.0, 1.0 - spread)

    return LiquidityDecision(
        allowed=True,
        reason="ok",
        execution_score=score,
    )