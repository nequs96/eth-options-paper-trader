# Q8-Q20 All-in-One Patch

This bundle adds the post-Q8 safety/validation roadmap in one package.

## Included

- Q8 Scheduler Risk Guard + optimizer-enforced guarded overnight runner
- Q9 Portfolio de-risking recommendations
- Q10 Historical snapshot store
- Q11 Historical replay framework
- Q12 Realistic execution/slippage simulation
- Q13 Risk-budget sizing hints
- Q14 Smart exit paper executor, default recommendation-only
- Q15 Alert log
- Q16 Morning report
- Q17 Compile/test runner
- Q18 Regime classifier/policy stub
- Q19 Exposure/concentration balancer
- Q20 Live-readiness safety layer, paper-only locked

## Install

```bash
python3 install_q8_q20_all_in_one_patch.py
```

Expected:

```text
Q8_Q20_AUDIT_OK
Q8_Q20_ALL_IN_ONE_PATCH_INSTALLED_OK
```

## Run full suite

```bash
python3 -m scripts.run_q8_q20_suite
```

## Safer overnight command after install

```bash
mkdir -p logs
caffeinate -dimsu python3 -u -m scripts.run_guarded_overnight 2>&1 | tee -a logs/guarded_overnight.log
```

This guarded runner checks risk and optimizer approval before entries.
