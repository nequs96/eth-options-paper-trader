"""
backtesting/backtest_engine.py

ETH options backtest engine.

This engine:
- downloads ETH historical price data
- estimates rolling historical volatility
- simulates market implied volatility
- prices ETH options with Black-Scholes
- detects cheap/expensive options
- applies risk rules
- opens/closes option positions
- tracks equity curve
- calculates performance metrics
- saves output CSV files

Important:
This is for research and education only.
It does not place real trades.
It is not financial advice.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from data.market_data import download_eth_data, get_close_prices
from models.black_scholes import black_scholes_price
from models.volatility import rolling_volatility
from strategies.option_mispricing import analyze_option_mispricing
from strategies.risk_rules import approve_trade
from backtesting.portfolio import Portfolio
from backtesting.metrics import calculate_performance_metrics, print_performance_metrics


@dataclass
class BacktestConfig:
    """
    Configuration for the ETH options backtest.
    """

    start_date: str = "2021-01-01"
    end_date: str | None = None
    interval: str = "1d"

    initial_cash: float = 10_000.0

    option_type: str = "call"
    option_dte_days: int = 30
    max_holding_days: int = 10

    risk_free_rate: float = 0.04

    rolling_vol_window: int = 30
    rolling_median_multiplier: int = 3

    max_risk_per_trade: float = 0.01

    price_threshold: float = 0.08
    volatility_threshold: float = 0.08

    take_profit_pct: float = 0.40
    stop_loss_pct: float = -0.30

    transaction_fee_rate: float = 0.001
    slippage_rate: float = 0.001

    min_volatility: float = 0.10
    max_volatility: float = 2.50

    strike_rounding: int = 50

    output_folder: str = "outputs"


@dataclass
class OpenPositionMeta:
    """
    Metadata for one open option position.
    """

    entry_index: int
    entry_date: pd.Timestamp
    entry_spot_price: float
    entry_option_price: float
    strike_price: float
    quantity: float
    original_dte_days: int


@dataclass
class BacktestResult:
    """
    Backtest result container.
    """

    equity_curve: pd.DataFrame
    trade_log: pd.DataFrame
    final_cash: float
    number_of_trades: int


def ensure_output_folder(folder: str) -> None:
    """
    Create output folder if it does not exist.
    """

    Path(folder).mkdir(parents=True, exist_ok=True)


def choose_atm_strike(
    spot_price: float,
    round_to: int,
) -> float:
    """
    Choose an approximate at-the-money strike.

    Example:
        spot_price = 3187
        round_to = 50
        strike = 3200
    """

    if spot_price <= 0:
        raise ValueError("spot_price must be greater than 0.")

    if round_to <= 0:
        raise ValueError("round_to must be greater than 0.")

    return float(round(spot_price / round_to) * round_to)


def simulate_market_implied_volatility(
    historical_volatility: float,
    rolling_median_volatility: float,
) -> float:
    """
    Simulate market implied volatility.

    Later this should be replaced by real ETH options implied volatility
    from an exchange such as Deribit.

    Logic:
    - If current realized volatility is low compared with median,
      simulated IV is discounted.
    - If current realized volatility is high compared with median,
      simulated IV is premium-priced.
    - Otherwise IV is close to historical volatility.
    """

    if historical_volatility <= 0:
        raise ValueError("historical_volatility must be greater than 0.")

    if rolling_median_volatility <= 0:
        return float(historical_volatility)

    vol_ratio = historical_volatility / rolling_median_volatility

    if vol_ratio < 0.80:
        iv_multiplier = 0.85
    elif vol_ratio > 1.20:
        iv_multiplier = 1.20
    else:
        iv_multiplier = 1.00

    simulated_iv = historical_volatility * iv_multiplier

    return float(max(simulated_iv, 0.0001))


def calculate_option_price(
    spot_price: float,
    strike_price: float,
    days_to_expiry: int,
    risk_free_rate: float,
    volatility: float,
    option_type: str,
) -> float:
    """
    Calculate option price using Black-Scholes.
    """

    if days_to_expiry <= 0:
        time_to_expiry = 0.0
    else:
        time_to_expiry = days_to_expiry / 365.0

    return black_scholes_price(
        S=spot_price,
        K=strike_price,
        T=time_to_expiry,
        r=risk_free_rate,
        sigma=volatility,
        option_type=option_type,
    )


def apply_entry_costs(
    option_price: float,
    fee_rate: float,
    slippage_rate: float,
) -> float:
    """
    Apply fee and slippage to a buy entry price.
    """

    if option_price < 0:
        raise ValueError("option_price cannot be negative.")

    return float(option_price * (1.0 + fee_rate + slippage_rate))


def apply_exit_costs(
    option_price: float,
    fee_rate: float,
    slippage_rate: float,
) -> float:
    """
    Apply fee and slippage to a sell exit price.
    """

    if option_price < 0:
        raise ValueError("option_price cannot be negative.")

    return float(max(option_price * (1.0 - fee_rate - slippage_rate), 0.0))


def should_close_position(
    holding_days: int,
    remaining_dte_days: int,
    entry_price: float,
    current_exit_price: float,
    config: BacktestConfig,
) -> tuple[bool, str]:
    """
    Decide whether an open position should be closed.
    """

    if entry_price <= 0:
        return True, "invalid_entry_price"

    pnl_pct = current_exit_price / entry_price - 1.0

    if pnl_pct >= config.take_profit_pct:
        return True, "take_profit"

    if pnl_pct <= config.stop_loss_pct:
        return True, "stop_loss"

    if holding_days >= config.max_holding_days:
        return True, "max_holding_days"

    if remaining_dte_days <= 1:
        return True, "near_expiry"

    return False, "hold"


def record_equity(
    equity_records: list[dict[str, Any]],
    timestamp: pd.Timestamp,
    portfolio: Portfolio,
    spot_price: float,
    historical_volatility: float | None,
    implied_volatility: float | None,
    mark_price: float | None,
) -> None:
    """
    Record one equity curve row.
    """

    if portfolio.number_of_open_positions() > 0 and mark_price is not None:
        equity = portfolio.equity([mark_price])
    else:
        equity = portfolio.equity()

    equity_records.append(
        {
            "timestamp": str(timestamp),
            "equity": float(equity),
            "cash": float(portfolio.cash),
            "spot_price": float(spot_price),
            "historical_volatility": historical_volatility,
            "simulated_implied_volatility": implied_volatility,
            "open_positions": portfolio.number_of_open_positions(),
        }
    )


def run_backtest(
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """
    Run the ETH options backtest.
    """

    if config is None:
        config = BacktestConfig()

    ensure_output_folder(config.output_folder)

    print("========== STARTING BACKTEST ==========")
    print(f"Start date:        {config.start_date}")
    print(f"End date:          {config.end_date}")
    print(f"Interval:          {config.interval}")
    print(f"Initial cash:      ${config.initial_cash:,.2f}")
    print(f"Option type:       {config.option_type.upper()}")
    print(f"Option DTE:        {config.option_dte_days} days")
    print("=======================================\n")

    print("Downloading ETH data...")

    data = download_eth_data(
        start_date=config.start_date,
        end_date=config.end_date,
        interval=config.interval,
    )

    close_prices = get_close_prices(data)

    data = data.copy()
    data["close"] = close_prices.values

    print(f"Downloaded rows: {len(data)}")

    vol_series = rolling_volatility(
        prices=close_prices,
        window=config.rolling_vol_window,
        timeframe=config.interval,
        use_log_returns=True,
    )

    rolling_median_window = config.rolling_vol_window * config.rolling_median_multiplier

    rolling_median_vol = vol_series.rolling(
        window=rolling_median_window,
        min_periods=config.rolling_vol_window,
    ).median()

    portfolio = Portfolio(initial_cash=config.initial_cash)

    open_meta: OpenPositionMeta | None = None

    equity_records: list[dict[str, Any]] = []
    trade_records: list[dict[str, Any]] = []

    start_index = rolling_median_window

    if start_index >= len(data):
        raise ValueError(
            "Not enough data for selected rolling windows. "
            "Use earlier start_date or smaller rolling_vol_window."
        )

    for i in range(start_index, len(data)):
        row = data.iloc[i]

        current_date = pd.to_datetime(row["timestamp"])
        spot_price = float(row["close"])

        raw_hist_vol = vol_series.iloc[i]
        raw_median_vol = rolling_median_vol.iloc[i]

        if pd.isna(raw_hist_vol) or pd.isna(raw_median_vol):
            record_equity(
                equity_records=equity_records,
                timestamp=current_date,
                portfolio=portfolio,
                spot_price=spot_price,
                historical_volatility=None,
                implied_volatility=None,
                mark_price=None,
            )
            continue

        historical_vol = float(raw_hist_vol)
        median_vol = float(raw_median_vol)

        if historical_vol <= 0 or median_vol <= 0:
            record_equity(
                equity_records=equity_records,
                timestamp=current_date,
                portfolio=portfolio,
                spot_price=spot_price,
                historical_volatility=None,
                implied_volatility=None,
                mark_price=None,
            )
            continue

        simulated_iv = simulate_market_implied_volatility(
            historical_volatility=historical_vol,
            rolling_median_volatility=median_vol,
        )

        current_mark_price: float | None = None

        # ==================================================
        # 1. Manage existing open position
        # ==================================================
        if open_meta is not None and portfolio.number_of_open_positions() > 0:
            holding_days = int(i - open_meta.entry_index)
            original_dte_days = int(open_meta.original_dte_days)
            remaining_dte_days = max(original_dte_days - holding_days, 0)

            current_market_price = calculate_option_price(
                spot_price=spot_price,
                strike_price=open_meta.strike_price,
                days_to_expiry=remaining_dte_days,
                risk_free_rate=config.risk_free_rate,
                volatility=simulated_iv,
                option_type=config.option_type,
            )

            current_exit_price = apply_exit_costs(
                option_price=current_market_price,
                fee_rate=config.transaction_fee_rate,
                slippage_rate=config.slippage_rate,
            )

            current_mark_price = current_exit_price

            close_now, close_reason = should_close_position(
                holding_days=holding_days,
                remaining_dte_days=remaining_dte_days,
                entry_price=open_meta.entry_option_price,
                current_exit_price=current_exit_price,
                config=config,
            )

            if close_now:
                trade = portfolio.close_position(
                    position_index=0,
                    exit_price=current_exit_price,
                )

                trade_records.append(
                    {
                        "entry_date": str(open_meta.entry_date),
                        "exit_date": str(current_date),
                        "option_type": config.option_type,
                        "entry_spot_price": float(open_meta.entry_spot_price),
                        "exit_spot_price": float(spot_price),
                        "strike_price": float(open_meta.strike_price),
                        "entry_option_price": float(open_meta.entry_option_price),
                        "exit_option_price": float(current_exit_price),
                        "quantity": float(open_meta.quantity),
                        "holding_days": int(holding_days),
                        "close_reason": close_reason,
                        "pnl": float(trade.pnl),
                    }
                )

                open_meta = None
                current_mark_price = None

        # ==================================================
        # 2. Look for new entry
        # ==================================================
        if open_meta is None and portfolio.number_of_open_positions() == 0:
            strike_price = choose_atm_strike(
                spot_price=spot_price,
                round_to=config.strike_rounding,
            )

            theoretical_price = calculate_option_price(
                spot_price=spot_price,
                strike_price=strike_price,
                days_to_expiry=config.option_dte_days,
                risk_free_rate=config.risk_free_rate,
                volatility=historical_vol,
                option_type=config.option_type,
            )

            simulated_market_price = calculate_option_price(
                spot_price=spot_price,
                strike_price=strike_price,
                days_to_expiry=config.option_dte_days,
                risk_free_rate=config.risk_free_rate,
                volatility=simulated_iv,
                option_type=config.option_type,
            )

            # Avoid invalid zero-priced options.
            if theoretical_price > 0 and simulated_market_price > 0:
                try:
                    mispricing = analyze_option_mispricing(
                        market_price=simulated_market_price,
                        spot_price=spot_price,
                        strike_price=strike_price,
                        time_to_expiry=config.option_dte_days / 365.0,
                        risk_free_rate=config.risk_free_rate,
                        historical_volatility=historical_vol,
                        option_type=config.option_type,
                        price_threshold=config.price_threshold,
                        volatility_threshold=config.volatility_threshold,
                    )
                except ValueError:
                    mispricing = None

                if mispricing is not None and mispricing.classification == "cheap":
                    entry_price = apply_entry_costs(
                        option_price=simulated_market_price,
                        fee_rate=config.transaction_fee_rate,
                        slippage_rate=config.slippage_rate,
                    )

                    risk_decision = approve_trade(
                        capital=portfolio.cash,
                        option_price=entry_price,
                        implied_volatility=mispricing.implied_volatility,
                        historical_volatility=mispricing.historical_volatility,
                        max_risk_per_trade=config.max_risk_per_trade,
                        max_volatility=config.max_volatility,
                        min_volatility=config.min_volatility,
                    )

                    if risk_decision.allowed and risk_decision.position_size > 0:
                        portfolio.open_position(
                            option_type=config.option_type,
                            entry_price=entry_price,
                            quantity=risk_decision.position_size,
                            strike_price=strike_price,
                            days_to_expiry=config.option_dte_days,
                            direction="long",
                        )

                        open_meta = OpenPositionMeta(
                            entry_index=int(i),
                            entry_date=current_date,
                            entry_spot_price=float(spot_price),
                            entry_option_price=float(entry_price),
                            strike_price=float(strike_price),
                            quantity=float(risk_decision.position_size),
                            original_dte_days=int(config.option_dte_days),
                        )

                        current_mark_price = entry_price

        # ==================================================
        # 3. Record equity for this timestep
        # ==================================================
        record_equity(
            equity_records=equity_records,
            timestamp=current_date,
            portfolio=portfolio,
            spot_price=spot_price,
            historical_volatility=historical_vol,
            implied_volatility=simulated_iv,
            mark_price=current_mark_price,
        )

    # ======================================================
    # 4. Force close any remaining open position at the end
    # ======================================================
    if open_meta is not None and portfolio.number_of_open_positions() > 0:
        final_row = data.iloc[-1]
        final_date = pd.to_datetime(final_row["timestamp"])
        final_spot_price = float(final_row["close"])

        raw_final_vol = vol_series.iloc[-1]

        if pd.isna(raw_final_vol) or float(raw_final_vol) <= 0:
            final_volatility = 0.75
        else:
            final_volatility = float(raw_final_vol)

        final_market_price = calculate_option_price(
            spot_price=final_spot_price,
            strike_price=open_meta.strike_price,
            days_to_expiry=1,
            risk_free_rate=config.risk_free_rate,
            volatility=final_volatility,
            option_type=config.option_type,
        )

        final_exit_price = apply_exit_costs(
            option_price=final_market_price,
            fee_rate=config.transaction_fee_rate,
            slippage_rate=config.slippage_rate,
        )

        trade = portfolio.close_position(
            position_index=0,
            exit_price=final_exit_price,
        )

        holding_days = int(len(data) - 1 - open_meta.entry_index)

        trade_records.append(
            {
                "entry_date": str(open_meta.entry_date),
                "exit_date": str(final_date),
                "option_type": config.option_type,
                "entry_spot_price": float(open_meta.entry_spot_price),
                "exit_spot_price": float(final_spot_price),
                "strike_price": float(open_meta.strike_price),
                "entry_option_price": float(open_meta.entry_option_price),
                "exit_option_price": float(final_exit_price),
                "quantity": float(open_meta.quantity),
                "holding_days": int(holding_days),
                "close_reason": "final_close",
                "pnl": float(trade.pnl),
            }
        )

        record_equity(
            equity_records=equity_records,
            timestamp=final_date,
            portfolio=portfolio,
            spot_price=final_spot_price,
            historical_volatility=final_volatility,
            implied_volatility=final_volatility,
            mark_price=None,
        )

    equity_curve = pd.DataFrame(equity_records)
    trade_log = pd.DataFrame(trade_records)

    if equity_curve.empty:
        raise ValueError("Backtest produced empty equity curve.")

    equity_series = equity_curve["equity"].astype(float)

    metrics = calculate_performance_metrics(
        equity_curve=equity_series,
        trades=portfolio.closed_trades,
        periods_per_year=365,
        risk_free_rate=config.risk_free_rate,
    )

    print()
    print_performance_metrics(metrics)

    equity_output_path = Path(config.output_folder) / "equity_curve.csv"
    trades_output_path = Path(config.output_folder) / "trade_log.csv"

    equity_curve.to_csv(equity_output_path, index=False)
    trade_log.to_csv(trades_output_path, index=False)

    print("\n========== BACKTEST OUTPUTS ==========")
    print(f"Equity curve saved to: {equity_output_path}")
    print(f"Trade log saved to:    {trades_output_path}")
    print(f"Closed trades:         {len(portfolio.closed_trades)}")
    print(f"Final cash:            ${portfolio.cash:,.2f}")
    print("======================================")

    return BacktestResult(
        equity_curve=equity_curve,
        trade_log=trade_log,
        final_cash=float(portfolio.cash),
        number_of_trades=len(portfolio.closed_trades),
    )


if __name__ == "__main__":
    backtest_config = BacktestConfig(
        start_date="2021-01-01",
        end_date=None,
        interval="1d",
        initial_cash=10_000.0,
        option_type="call",
        option_dte_days=30,
        max_holding_days=10,
        risk_free_rate=0.04,
        rolling_vol_window=30,
        max_risk_per_trade=0.01,
        price_threshold=0.08,
        volatility_threshold=0.08,
        take_profit_pct=0.40,
        stop_loss_pct=-0.30,
        transaction_fee_rate=0.001,
        slippage_rate=0.001,
        min_volatility=0.10,
        max_volatility=2.50,
        strike_rounding=50,
        output_folder="outputs",
    )

    run_backtest(backtest_config)