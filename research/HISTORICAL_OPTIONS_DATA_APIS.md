# Historical options data providers for 6-12 month ETH Deribit backtesting

Preferred data shape:

```text
timestamp, instrument_name, option_type, expiry, strike, underlying_price_usd,
bid_price_usd, ask_price_usd, mark_price_usd, implied_volatility,
delta, gamma, vega, theta, open_interest, volume
```

Supported adapter status:

- `local_csv`: production-ready for any exported provider CSV after normalization.
- `laevitas`: direct snapshot endpoint scaffold using `LAEVITAS_API_KEY`.
- `tardis`: raw NDJSON/ticker scaffold using `TARDIS_API_KEY`; full production normalization may require provider message-specific mapping.
- `coinapi`, `amberdata`, `cryptodatadownload`: use exported CSVs through `local_csv` unless subscription-specific endpoint paths are added.

## Fetch / normalize

```bash
python3 -m scripts.fetch_historical_options --provider local_csv --start 2025-11-01 --end 2026-05-01 --output-dir data/historical_options
```

## Run backtest

```bash
python3 -m scripts.run_6m_backtest --input-file data/historical_options/normalized.csv
python3 -m scripts.run_12m_backtest --input-file data/historical_options/normalized.csv
```
