from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class AutonomousSafetyConfig:
    max_open_positions: int = 8
    max_risk_breach_cycles: int = 2
    max_error_cycles: int = 2
    max_drawdown_pct: float = -0.05
    max_abs_net_theta: float = 50.0
    max_flat_7d_loss_abs: float = 350.0
    pause_after_breach_minutes: int = 60
    resume_required_clean_cycles: int = 2
    stale_file_minutes: int = 30
    max_auto_derisk_per_cycle: int = 3
    allow_paper_auto_derisk: bool = True
    allow_paper_auto_exit: bool = True
    real_trading_allowed: bool = False
