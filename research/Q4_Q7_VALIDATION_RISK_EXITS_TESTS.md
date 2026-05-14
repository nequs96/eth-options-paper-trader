# Q4-Q7 Validation, Risk, Exits, and Tests Patch

Adds Q4 walk-forward/parameter stability, Q5 portfolio risk limits, Q6 smart exit recommendations, and Q7 audit/test/dashboard support.

## Install

```bash
python install_q4_q7_patch.py
```

## Audit

```bash
python -m scripts.audit_q4_q7
```

## Run

```bash
python -m scripts.run_q4_q7_suite
```

Expected final line:

```text
Q4_Q7_SUITE_COMPLETE
```

## Outputs

```text
outputs/walk_forward_results.csv
outputs/walk_forward_summary.csv
outputs/parameter_stability_report.csv
outputs/portfolio_limit_report.csv
outputs/risk_limit_breaches.csv
outputs/smart_exit_recommendations.csv
outputs/enhanced_research_dashboard.html
```
