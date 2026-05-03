"""
config.py

Central configuration file for the ETH options algorithm.

This file stores default parameters used by:
- Black-Scholes pricing
- Greeks calculation
- Payoff visualization
- Risk management
- Future strategy modules

Important:
These values are only defaults. They are not trading advice.
You should later replace assumptions with real market data.
"""

# ============================================================
# Asset settings
# ============================================================

UNDERLYING_SYMBOL = "ETH"
QUOTE_CURRENCY = "USD"
TRADING_PAIR = "ETH/USD"


# ============================================================
# Default option contract assumptions
# ============================================================

# Current ETH price assumption
# Later this will come from live/historical market data.
DEFAULT_SPOT_PRICE = 3000.0

# Option strike price
DEFAULT_STRIKE_PRICE = 3200.0

# Time to expiry in days
DEFAULT_DAYS_TO_EXPIRY = 30

# Convert days to years for Black-Scholes
DAYS_IN_YEAR = 365.0

# Option type: "call" or "put"
DEFAULT_OPTION_TYPE = "call"


# ============================================================
# Market assumptions
# ============================================================

# Risk-free rate as decimal
# Example: 0.04 means 4% annual risk-free rate.
RISK_FREE_RATE = 0.04

# Default annualized ETH volatility
# Example: 0.75 means 75% annual volatility.
DEFAULT_VOLATILITY = 0.75

# Dividend yield is normally 0 for ETH spot.
# For some advanced models, this may be replaced with staking yield,
# funding rate, or carry adjustment.
DIVIDEND_YIELD = 0.0


# ============================================================
# Visualization settings
# ============================================================

# Price range for payoff and surface plots
MIN_ETH_PRICE = 1000.0
MAX_ETH_PRICE = 6000.0
PRICE_STEPS = 200

# Expiry range for 3D surface plots
MIN_DAYS_TO_EXPIRY = 1
MAX_DAYS_TO_EXPIRY = 180
EXPIRY_STEPS = 80

# Volatility range for volatility sensitivity plots
MIN_VOLATILITY = 0.20
MAX_VOLATILITY = 1.50
VOLATILITY_STEPS = 100


# ============================================================
# Trading and risk assumptions
# ============================================================

INITIAL_CAPITAL = 10_000.0

# Maximum percentage of capital risked per trade.
# Example: 0.01 means 1%.
MAX_RISK_PER_TRADE = 0.01

# Maximum allowed portfolio drawdown before stopping trading.
# Example: 0.15 means 15%.
MAX_DRAWDOWN = 0.15

# Transaction cost assumption.
# Example: 0.001 means 0.1%.
TRANSACTION_FEE_RATE = 0.001

# Slippage assumption.
# Example: 0.001 means 0.1%.
SLIPPAGE_RATE = 0.001


# ============================================================
# Strategy assumptions
# ============================================================

# If market option price is this much above theoretical value,
# we may consider it expensive.
MISPRICING_THRESHOLD = 0.10

# Example:
# If theoretical price is $100 and market price is $115,
# mispricing = 15%, so it passes the 10% threshold.


# ============================================================
# Data settings for later use
# ============================================================

DATA_FOLDER = "data"
OUTPUT_FOLDER = "outputs"

# Default timeframe for ETH market data
DEFAULT_TIMEFRAME = "1h"

# Historical data range for future backtesting
DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE = "2026-01-01"


# ============================================================
# Helper functions
# ============================================================

def days_to_years(days: float) -> float:
    """
    Convert days to years for Black-Scholes calculations.
    """
    return days / DAYS_IN_YEAR


def get_default_time_to_expiry() -> float:
    """
    Return default time to expiry in years.
    """
    return days_to_years(DEFAULT_DAYS_TO_EXPIRY)


def print_config_summary() -> None:
    """
    Print the current configuration summary.
    Useful for checking assumptions before running the algorithm.
    """
    print("========== ETH Options Algorithm Configuration ==========")
    print(f"Underlying: {UNDERLYING_SYMBOL}")
    print(f"Trading pair: {TRADING_PAIR}")
    print(f"Default spot price: {DEFAULT_SPOT_PRICE}")
    print(f"Default strike price: {DEFAULT_STRIKE_PRICE}")
    print(f"Default days to expiry: {DEFAULT_DAYS_TO_EXPIRY}")
    print(f"Default time to expiry in years: {get_default_time_to_expiry():.4f}")
    print(f"Default option type: {DEFAULT_OPTION_TYPE}")
    print(f"Risk-free rate: {RISK_FREE_RATE:.2%}")
    print(f"Default volatility: {DEFAULT_VOLATILITY:.2%}")
    print(f"Initial capital: {INITIAL_CAPITAL}")
    print(f"Max risk per trade: {MAX_RISK_PER_TRADE:.2%}")
    print(f"Max drawdown: {MAX_DRAWDOWN:.2%}")
    print("=========================================================")