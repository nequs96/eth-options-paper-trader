# Institutional ETH Options Research Map

This is the consolidated architecture map for the research-grade ETH options paper-trading system.

## 1. Data Integrity
- Validate option-chain snapshots.
- Validate candidate tables.
- Preserve reject reasons.
- Track empty files and missing schemas.

## 2. Candidate Engine
- Rank relevant strikes/maturities first.
- Score edge, volatility, liquidity, Greeks, regime, and portfolio fit.
- Save accepted and rejected rows.

## 3. Portfolio Construction
- Prevent duplicate instruments.
- Respect cash buffer and open-risk budget.
- Size dynamically from confidence and drawdown.

## 4. Position Management
- Mark to market.
- Track unrealized PnL, highest price, trailing stops.
- Close into trade history.

## 5. Institutional Database
- Store option snapshots, quotes, candidates, rejected rows, account snapshots, position snapshots, and system events.

## 6. Reporting
- Keep terminal clean.
- Keep full research detail in CSV and SQLite.
- Generate cycle summaries and diagnostics.

## 7. Future Quant Frontier
- SVI/SABR vol surface fitting.
- Scenario Greeks.
- Monte Carlo stress.
- Walk-forward validation.
- Regime-specific parameter validation.
