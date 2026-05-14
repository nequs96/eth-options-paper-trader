# Q21-Q25 Autonomous Safety Pack

This pack prepares the paper system for unattended autonomous paper operation by adding:

- Q21 Global Safety Kernel
- Q22 Autonomous paper de-risk executor
- Q23 Regime failure detector
- Q24 Pause/resume controller
- Q25 Self-termination guard

The pack is paper-only. It does not permit real-money trading.

## Install

```bash
python3 install_q21_q25_autonomous_safety_pack.py
```

## Audit

```bash
python3 -m scripts.audit_autonomous_safety
```

## Run autonomous paper daemon

```bash
mkdir -p logs
caffeinate -dimsu python3 -u -m scripts.autonomous_paper_daemon 2>&1 | tee -a logs/autonomous_paper_daemon.log
```

## Emergency safe mode

```bash
python3 -m scripts.safe_mode
```

Safe mode activates kill switch, updates reports, attempts paper de-risking, and writes a morning report.
