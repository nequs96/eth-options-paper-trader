from dataclasses import dataclass
@dataclass
class LiquidityDecision:
    allowed: bool; reason: str; execution_score: float
def safe_float(x):
    try: return float(x)
    except Exception: return None
def evaluate_liquidity(market_price_usd,bid_ask_spread_pct,open_interest=None,min_price_usd=10.0,max_spread_pct=0.20):
    p=safe_float(market_price_usd); s=safe_float(bid_ask_spread_pct)
    if p is None or p<=0: return LiquidityDecision(False,'invalid_price',0)
    if p<min_price_usd: return LiquidityDecision(False,'price_too_low',.1)
    if s is not None and s>max_spread_pct: return LiquidityDecision(False,'spread_too_wide',.2)
    return LiquidityDecision(True,'ok',.8)
