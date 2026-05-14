# Q32 Profitability + Robustness Patch

This patch is paper-only. It cannot guarantee profitability. It adds stricter loss control, profit locking, call throttling, forced de-risk recommendations, and a harder candidate filter.

## Install

```bash
python3 install_q32_profitability_robustness_patch.py
```

## Audit

```bash
python3 -m scripts.audit_q32_profitability
```

## Recommendation-only cycle

```bash
python3 -m scripts.run_q32_profitability_suite
```

## Management-only with paper execution of profit-lock/derisk closes

```bash
python3 -m scripts.run_q32_management_only_cycle --execute
```

## Guarded loop, recommendation-only

```bash
caffeinate -dimsu python3 -u -m scripts.run_q32_guarded_paper_loop 2>&1 | tee -a logs/q32_guarded_paper_loop.log
```

## Guarded loop, paper auto-derisk enabled

```bash
caffeinate -dimsu python3 -u -m scripts.run_q32_guarded_paper_loop --execute-derisk 2>&1 | tee -a logs/q32_guarded_paper_loop.log
```
