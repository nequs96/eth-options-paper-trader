"""
main.py

Main entry point for the ETH options algorithm.

This script:
1. Loads configuration values
2. Downloads ETH historical market data
3. Estimates ETH historical volatility
4. Prices an ETH option using Black-Scholes
5. Calculates Greeks
6. Calculates implied volatility from an example market price
7. Runs basic option mispricing research signal
8. Runs risk management approval
9. Generates payoff visualization
10. Generates 3D option price surface

Important:
This project is for research and education. It is not financial advice.
"""

from config import (
    DEFAULT_STRIKE_PRICE,
    DEFAULT_DAYS_TO_EXPIRY,
    DEFAULT_OPTION_TYPE,
    RISK_FREE_RATE,
    DEFAULT_VOLATILITY,
    MIN_ETH_PRICE,
    MAX_ETH_PRICE,
    PRICE_STEPS,
    MIN_DAYS_TO_EXPIRY,
    MAX_DAYS_TO_EXPIRY,
    DAYS_IN_YEAR,
    DEFAULT_START_DATE,
    DEFAULT_END_DATE,
    INITIAL_CAPITAL,
    MAX_RISK_PER_TRADE,
    days_to_years,
    print_config_summary,
)

from data.market_data import (
    download_eth_data,
    get_close_prices,
    get_latest_eth_price,
)

from models.black_scholes import black_scholes_price
from models.greeks import all_greeks
from models.volatility import historical_volatility, summarize_volatility
from models.implied_volatility import implied_volatility_summary

from strategies.option_mispricing import (
    MispricingResult,
    analyze_option_mispricing,
    print_mispricing_result,
)

from strategies.risk_rules import (
    approve_trade,
    print_risk_decision,
)

from visualization.payoff_plots import plot_option_payoff
from visualization.surface_plots import plot_option_price_surface


def load_real_eth_assumptions() -> tuple[float, float]:
    """
    Download ETH data and estimate:
    - latest ETH spot price
    - annualized historical volatility

    Returns
    -------
    tuple[float, float]
        Latest ETH price and estimated annualized volatility.
    """

    print("Downloading ETH market data...")

    try:
        eth_data = download_eth_data(
            start_date=DEFAULT_START_DATE,
            end_date=DEFAULT_END_DATE,
            interval="1d",
        )

        close_prices = get_close_prices(eth_data)
        latest_spot_price = get_latest_eth_price(eth_data)

        estimated_volatility = historical_volatility(
            prices=close_prices,
            timeframe="1d",
            use_log_returns=True,
        )

        vol_summary = summarize_volatility(
            prices=close_prices,
            timeframe="1d",
            rolling_window=30,
        )

        print("\n========== REAL ETH DATA SUMMARY ==========")
        print(f"Rows downloaded: {len(eth_data)}")
        print(f"Latest ETH close: ${latest_spot_price:,.2f}")
        print(f"Full historical volatility: {estimated_volatility:.2%}")
        print(f"Latest 30D rolling volatility: {vol_summary['latest_rolling_volatility']:.2%}")
        print(f"Volatility rank: {vol_summary['latest_volatility_rank']:.2%}")
        print("===========================================\n")

        return float(latest_spot_price), float(estimated_volatility)

    except Exception as error:
        print("\nWARNING: Could not load real ETH data.")
        print(f"Reason: {error}")
        print("Falling back to config default assumptions.\n")

        from config import DEFAULT_SPOT_PRICE

        return float(DEFAULT_SPOT_PRICE), float(DEFAULT_VOLATILITY)


def run_pricing_example(
    spot_price: float,
    volatility: float,
) -> float:
    """
    Run Black-Scholes pricing and Greeks calculation.

    Parameters
    ----------
    spot_price : float
        ETH spot price.
    volatility : float
        Annualized ETH volatility.

    Returns
    -------
    float
        Theoretical option price.
    """

    T = days_to_years(DEFAULT_DAYS_TO_EXPIRY)

    option_price = black_scholes_price(
        S=spot_price,
        K=DEFAULT_STRIKE_PRICE,
        T=T,
        r=RISK_FREE_RATE,
        sigma=volatility,
        option_type=DEFAULT_OPTION_TYPE,
    )

    greeks = all_greeks(
        S=spot_price,
        K=DEFAULT_STRIKE_PRICE,
        T=T,
        r=RISK_FREE_RATE,
        sigma=volatility,
        option_type=DEFAULT_OPTION_TYPE,
    )

    print("\n========== ETH OPTION PRICING ==========")
    print(f"Option type:       {DEFAULT_OPTION_TYPE.upper()}")
    print(f"Spot price (ETH):  ${spot_price:,.2f}")
    print(f"Strike price:      ${DEFAULT_STRIKE_PRICE:,.2f}")
    print(f"Days to expiry:    {DEFAULT_DAYS_TO_EXPIRY}")
    print(f"Time to expiry:    {T:.4f} years")
    print(f"Risk-free rate:    {RISK_FREE_RATE:.2%}")
    print(f"Volatility used:   {volatility:.2%}")
    print("----------------------------------------")
    print(f"Theoretical price: ${option_price:,.4f}")
    print("========================================\n")

    print("============== GREEKS ==================")
    for name, value in greeks.items():
        print(f"{name:<20}: {value:>12.6f}")
    print("========================================\n")

    return float(option_price)


def run_implied_volatility_example(
    spot_price: float,
    historical_volatility_value: float,
    theoretical_price: float,
) -> None:
    """
    Run implied volatility example.

    For now, market option price is simulated as 10% above theoretical price.
    Later this should come from real ETH options market data.
    """

    T = days_to_years(DEFAULT_DAYS_TO_EXPIRY)

    simulated_market_price = theoretical_price * 1.10

    summary = implied_volatility_summary(
        market_price=simulated_market_price,
        S=spot_price,
        K=DEFAULT_STRIKE_PRICE,
        T=T,
        r=RISK_FREE_RATE,
        historical_vol=historical_volatility_value,
        option_type=DEFAULT_OPTION_TYPE,
        threshold=0.10,
    )

    print("========== IMPLIED VOLATILITY SUMMARY ==========")
    print(f"Simulated market option price: ${simulated_market_price:,.4f}")
    print("-----------------------------------------------")

    for key, value in summary.items():
        if isinstance(value, float):
            if "percent" in key:
                print(f"{key:<35}: {value:.2f}%")
            else:
                print(f"{key:<35}: {value:.6f}")
        else:
            print(f"{key:<35}: {value}")

    print("================================================\n")


def run_mispricing_strategy_example(
    spot_price: float,
    historical_volatility_value: float,
    theoretical_price: float,
) -> MispricingResult:
    """
    Run basic option mispricing research signal.

    For now, market price is simulated as 10% above theoretical price.
    Later this should come from real ETH options market data.

    Returns
    -------
    MispricingResult
        Full mispricing analysis result.
    """

    T = days_to_years(DEFAULT_DAYS_TO_EXPIRY)

    simulated_market_price = theoretical_price * 1.10

    result = analyze_option_mispricing(
        market_price=simulated_market_price,
        spot_price=spot_price,
        strike_price=DEFAULT_STRIKE_PRICE,
        time_to_expiry=T,
        risk_free_rate=RISK_FREE_RATE,
        historical_volatility=historical_volatility_value,
        option_type=DEFAULT_OPTION_TYPE,
        price_threshold=0.10,
        volatility_threshold=0.10,
    )

    print_mispricing_result(result)

    return result


def run_risk_management_example(
    mispricing_result: MispricingResult,
) -> None:
    """
    Run risk management approval based on mispricing result.

    The option market price is treated as the premium.
    For a long option, premium paid is the maximum loss.
    """

    decision = approve_trade(
        capital=INITIAL_CAPITAL,
        option_price=mispricing_result.market_price,
        implied_volatility=mispricing_result.implied_volatility,
        historical_volatility=mispricing_result.historical_volatility,
        max_risk_per_trade=MAX_RISK_PER_TRADE,
        max_volatility=2.0,
        min_volatility=0.10,
    )

    print_risk_decision(decision)


def run_payoff_visualization(
    option_price: float,
) -> None:
    """
    Generate payoff visualization using theoretical option price as premium.
    """

    print("Generating payoff chart...")

    plot_option_payoff(
        strike_price=DEFAULT_STRIKE_PRICE,
        premium_paid=option_price,
        option_type=DEFAULT_OPTION_TYPE,
        min_price=MIN_ETH_PRICE,
        max_price=MAX_ETH_PRICE,
        steps=PRICE_STEPS,
        show_plot=True,
    )


def run_surface_visualization(
    volatility: float,
) -> None:
    """
    Generate 3D Black-Scholes option price surface.
    """

    print("Generating 3D option price surface...")

    plot_option_price_surface(
        strike_price=DEFAULT_STRIKE_PRICE,
        risk_free_rate=RISK_FREE_RATE,
        volatility=volatility,
        option_type=DEFAULT_OPTION_TYPE,
        min_price=MIN_ETH_PRICE,
        max_price=MAX_ETH_PRICE,
        price_steps=80,
        min_days=MIN_DAYS_TO_EXPIRY,
        max_days=MAX_DAYS_TO_EXPIRY,
        expiry_steps=60,
        days_in_year=DAYS_IN_YEAR,
        show_plot=True,
    )


def main() -> None:
    """
    Main execution function.
    """

    print_config_summary()

    spot_price, estimated_volatility = load_real_eth_assumptions()

    theoretical_price = run_pricing_example(
        spot_price=spot_price,
        volatility=estimated_volatility,
    )

    run_implied_volatility_example(
        spot_price=spot_price,
        historical_volatility_value=estimated_volatility,
        theoretical_price=theoretical_price,
    )

    mispricing_result = run_mispricing_strategy_example(
        spot_price=spot_price,
        historical_volatility_value=estimated_volatility,
        theoretical_price=theoretical_price,
    )

    run_risk_management_example(
        mispricing_result=mispricing_result,
    )

    run_payoff_visualization(
        option_price=theoretical_price,
    )

    run_surface_visualization(
        volatility=estimated_volatility,
    )


if __name__ == "__main__":
    main()