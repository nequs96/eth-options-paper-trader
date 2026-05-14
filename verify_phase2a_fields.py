from execution.live_scheduler import LiveSchedulerConfig
required = ['max_option_chain_instruments', 'ticker_timeout_seconds', 'option_chain_progress_every', 'save_partial_option_chain', 'run_healthcheck', 'strict_healthcheck']
fields = sorted(LiveSchedulerConfig.__dataclass_fields__.keys())
print(fields)
missing = [x for x in required if x not in fields]
if missing:
    raise SystemExit(f'MISSING: {missing}')
print('PHASE2A_FIELDS_OK')
