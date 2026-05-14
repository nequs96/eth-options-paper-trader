# Q1 Signal Quality + Risk-Gated Optimizer Patch

This patch tightens the optimizer so it stops selecting trades that are attractive only because they are very short-dated, thinly traded, or too expensive to execute.

## New hard gates

Default optimizer constraints:

```text
min_days_to_expiry = 7
max_days_to_expiry = 35
max_abs_moneyness = 0.12
min_open_interest = 100
min_volume = 20
max_spread_front_week = 0.06
max_spread_front_month = 0.08
max_spread_other = 0.10
min_institutional_edge_score = 0.45
max_abs_net_theta_after_hint = 50
max_flat_7d_loss_abs = 350
```

## New outputs

```text
outputs/optimized_trade_list.csv
outputs/optimizer_rejected_candidates.csv
outputs/optimizer_risk_gate_report.csv
```

## Why this was added

The prior optimizer selected very short-dated calls with about 3.36 DTE, including one contract with only 2 open interest and 2 volume. This patch makes the optimizer more conservative by default.

## Install

```bash
python install_q1_signal_quality_optimizer_patch.py
```

## Audit

```bash
python -m scripts.audit_q1_optimizer
```

## Run

After running the scheduler and research suite inputs, run:

```bash
python -m scripts.run_institutional_research_suite
```

or only rerun the optimizer:

```bash
python -m execution.portfolio_optimizer
```
