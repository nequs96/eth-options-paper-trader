from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Q32ProfitRobustConfig:
    max_open_positions: int = 4
    target_positions: int = 2
    max_new_positions_per_cycle: int = 1
    max_same_expiry_positions: int = 2
    max_same_option_type_positions: int = 3
    max_total_drawdown_abs: float = 350.0
    max_realized_loss_abs: float = 250.0
    max_rolling_closed_trade_loss_abs: float = 175.0
    rolling_trade_window: int = 8
    max_abs_net_theta: float = 35.0
    max_abs_net_vega: float = 45.0
    max_flat_7d_loss_abs: float = 250.0
    min_edge_score: float = 0.55
    min_surface_score: float = 0.62
    min_dte: float = 7.0
    max_dte: float = 30.0
    max_abs_moneyness: float = 0.10
    max_spread: float = 0.05
    min_open_interest: float = 500.0
    min_volume: float = 50.0
    profit_lock_trigger_pct: float = 0.25
    profit_take_partial_pct: float = 0.50
    profit_take_full_pct: float = 0.70
    profit_giveback_max_pct: float = 0.35
    min_profit_to_derisk_pct: float = 0.20
    call_loss_window: int = 8
    call_loss_rate_block_threshold: float = 0.60
    call_expectancy_block_threshold: float = -0.05
    call_block_hours: float = 24.0
    stale_file_minutes: int = 30
    allow_refresh_before_derisk: bool = True
