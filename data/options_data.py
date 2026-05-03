"""
data/options_data.py

Real ETH options market data from Deribit public API.

This module:
- fetches ETH option instruments
- fetches ticker data for selected options
- builds a live ETH option chain
- converts option prices from ETH units to USD
- saves option chain data to CSV

Important:
This file uses public market data only.
It does not require API keys.
It does not place trades.
It is for research and education only.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


DERIBIT_PRODUCTION_URL = "https://www.deribit.com/api/v2"
DERIBIT_TESTNET_URL = "https://test.deribit.com/api/v2"


@dataclass
class DeribitConfig:
    """
    Configuration for Deribit public data.
    """

    currency: str = "ETH"
    kind: str = "option"
    testnet: bool = False
    request_sleep_seconds: float = 0.15
    output_folder: str = "outputs"


def get_base_url(testnet: bool = False) -> str:
    """
    Return Deribit API base URL.
    """

    if testnet:
        return DERIBIT_TESTNET_URL

    return DERIBIT_PRODUCTION_URL


def safe_float(value: Any) -> float | None:
    """
    Convert value to float safely.

    Returns None if conversion is impossible.
    """

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_positive_float(value: Any) -> float | None:
    """
    Convert value to positive float safely.

    Returns None if value is missing, invalid, or negative.
    """

    number = safe_float(value)

    if number is None:
        return None

    if number < 0:
        return None

    return float(number)


def safe_timestamp_from_ms(value: Any) -> pd.Timestamp | None:
    """
    Convert millisecond timestamp to pandas Timestamp safely.
    """

    timestamp_value = safe_float(value)

    if timestamp_value is None:
        return None

    return pd.to_datetime(timestamp_value, unit="ms", utc=True)


def safe_datetime(value: Any) -> pd.Timestamp | None:
    """
    Convert value to pandas Timestamp safely.
    """

    if value is None:
        return None

    try:
        timestamp = pd.to_datetime(value, utc=True)
    except (TypeError, ValueError):
        return None

    if pd.isna(timestamp):
        return None

    return pd.Timestamp(timestamp)


def deribit_public_get(
    method: str,
    params: dict[str, Any] | None = None,
    testnet: bool = False,
    timeout: int = 15,
) -> dict[str, Any]:
    """
    Call Deribit public HTTP endpoint.

    Example:
        method = "public/get_instruments"
        params = {"currency": "ETH", "kind": "option", "expired": "false"}
    """

    if params is None:
        params = {}

    base_url = get_base_url(testnet=testnet)

    query = urllib.parse.urlencode(params)
    url = f"{base_url}/{method}"

    if query:
        url = f"{url}?{query}"

    request = urllib.request.Request(
        url=url,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "eth-options-research-bot/1.0",
        },
        method="GET",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw_response = response.read().decode("utf-8")
    except Exception as error:
        raise RuntimeError(f"Deribit request failed: {url}. Reason: {error}") from error

    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Could not decode Deribit JSON response: {raw_response}"
        ) from error

    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected Deribit response type: {type(payload)}")

    if "error" in payload:
        raise RuntimeError(f"Deribit API error: {payload['error']}")

    if "result" not in payload:
        raise RuntimeError(f"Unexpected Deribit response: {payload}")

    return payload


def fetch_eth_option_instruments(
    config: DeribitConfig | None = None,
    expired: bool = False,
) -> pd.DataFrame:
    """
    Fetch ETH option instruments from Deribit.

    Returns
    -------
    pd.DataFrame
        Available ETH option instruments.
    """

    if config is None:
        config = DeribitConfig()

    payload = deribit_public_get(
        method="public/get_instruments",
        params={
            "currency": config.currency,
            "kind": config.kind,
            "expired": str(expired).lower(),
        },
        testnet=config.testnet,
    )

    instruments = payload["result"]

    if not isinstance(instruments, list):
        raise RuntimeError("Deribit get_instruments result is not a list.")

    if len(instruments) == 0:
        return pd.DataFrame()

    data = pd.DataFrame(instruments)

    if data.empty:
        return data

    if "expiration_timestamp" in data.columns:
        data["expiration_datetime"] = pd.to_datetime(
            data["expiration_timestamp"],
            unit="ms",
            utc=True,
            errors="coerce",
        )

    if "creation_timestamp" in data.columns:
        data["creation_datetime"] = pd.to_datetime(
            data["creation_timestamp"],
            unit="ms",
            utc=True,
            errors="coerce",
        )

    if "strike" in data.columns:
        data["strike"] = pd.to_numeric(data["strike"], errors="coerce")

    if "is_active" in data.columns:
        data = data[data["is_active"] == True].copy()  # noqa: E712

    sort_columns = []

    if "expiration_datetime" in data.columns:
        sort_columns.append("expiration_datetime")

    if "strike" in data.columns:
        sort_columns.append("strike")

    if "option_type" in data.columns:
        sort_columns.append("option_type")

    if sort_columns:
        data = data.sort_values(
            by=sort_columns,
            ascending=True,
        ).reset_index(drop=True)

    return data


def fetch_deribit_ticker(
    instrument_name: str,
    config: DeribitConfig | None = None,
) -> dict[str, Any]:
    """
    Fetch Deribit ticker for one instrument.
    """

    if config is None:
        config = DeribitConfig()

    if not instrument_name:
        raise ValueError("instrument_name cannot be empty.")

    payload = deribit_public_get(
        method="public/ticker",
        params={"instrument_name": instrument_name},
        testnet=config.testnet,
    )

    result = payload["result"]

    if not isinstance(result, dict):
        raise RuntimeError("Deribit ticker result is not a dictionary.")

    return result


def fetch_eth_underlying_price(
    config: DeribitConfig | None = None,
) -> float:
    """
    Fetch ETH underlying/index price using ETH-PERPETUAL ticker.
    """

    if config is None:
        config = DeribitConfig()

    ticker = fetch_deribit_ticker(
        instrument_name="ETH-PERPETUAL",
        config=config,
    )

    price_candidates = [
        ticker.get("index_price"),
        ticker.get("estimated_delivery_price"),
        ticker.get("last_price"),
        ticker.get("mark_price"),
    ]

    for candidate in price_candidates:
        price = safe_float(candidate)

        if price is not None and price > 0:
            return float(price)

    raise RuntimeError("Could not get valid ETH underlying price from Deribit ticker.")


def normalize_deribit_iv(iv_value: Any) -> float | None:
    """
    Normalize Deribit IV value to decimal form.

    Example:
        75.0 -> 0.75
        0.75 -> 0.75
    """

    iv = safe_float(iv_value)

    if iv is None:
        return None

    if iv <= 0:
        return None

    if iv > 5.0:
        return float(iv / 100.0)

    return float(iv)


def option_price_to_usd(
    option_price_in_eth: Any,
    underlying_price_usd: float,
) -> float | None:
    """
    Convert Deribit ETH option premium into USD.

    Approximation:
        option_price_usd = option_price_in_eth * ETH underlying price
    """

    option_price = safe_positive_float(option_price_in_eth)

    if option_price is None:
        return None

    if underlying_price_usd <= 0:
        return None

    return float(option_price * underlying_price_usd)


def extract_greek(
    ticker: dict[str, Any],
    greek_name: str,
) -> float | None:
    """
    Extract Greek value from Deribit ticker if present.
    """

    greeks = ticker.get("greeks")

    if not isinstance(greeks, dict):
        return None

    return safe_float(greeks.get(greek_name))


def calculate_days_to_expiry(
    expiration_value: Any,
) -> float | None:
    """
    Calculate days to expiry from expiration datetime.
    """

    expiration_timestamp = safe_datetime(expiration_value)

    if expiration_timestamp is None:
        return None

    now_utc = pd.Timestamp.now(tz="UTC")

    days = (expiration_timestamp - now_utc).total_seconds() / 86400.0

    return float(max(days, 0.0))


def ticker_to_option_row(
    instrument: pd.Series,
    ticker: dict[str, Any],
    fallback_underlying_price: float,
) -> dict[str, Any]:
    """
    Convert instrument metadata and ticker data into one clean option-chain row.
    """

    underlying_candidates = [
        ticker.get("underlying_price"),
        ticker.get("index_price"),
        fallback_underlying_price,
    ]

    underlying_price_float: float | None = None

    for candidate in underlying_candidates:
        candidate_float = safe_float(candidate)

        if candidate_float is not None and candidate_float > 0:
            underlying_price_float = candidate_float
            break

    if underlying_price_float is None:
        raise RuntimeError("Could not determine underlying price for option row.")

    instrument_name = instrument.get("instrument_name")
    base_currency = instrument.get("base_currency")
    quote_currency = instrument.get("quote_currency")
    settlement_currency = instrument.get("settlement_currency")
    option_type = instrument.get("option_type")
    strike_value = safe_float(instrument.get("strike"))

    if strike_value is None:
        raise RuntimeError(f"Invalid strike for instrument: {instrument_name}")

    expiration_value = instrument.get("expiration_datetime")
    expiration_timestamp = safe_datetime(expiration_value)
    days_to_expiry = calculate_days_to_expiry(expiration_value)

    best_bid = ticker.get("best_bid_price")
    best_ask = ticker.get("best_ask_price")
    mark_price = ticker.get("mark_price")
    last_price = ticker.get("last_price")

    timestamp_value = ticker.get("timestamp")
    ticker_timestamp = safe_timestamp_from_ms(timestamp_value)

    return {
        "instrument_name": str(instrument_name),
        "base_currency": str(base_currency),
        "quote_currency": str(quote_currency),
        "settlement_currency": str(settlement_currency),
        "option_type": str(option_type),
        "strike": float(strike_value),
        "expiration_datetime": str(expiration_timestamp)
        if expiration_timestamp is not None
        else None,
        "days_to_expiry": days_to_expiry,
        "underlying_price": float(underlying_price_float),
        "best_bid_price_eth": safe_float(best_bid),
        "best_ask_price_eth": safe_float(best_ask),
        "mark_price_eth": safe_float(mark_price),
        "last_price_eth": safe_float(last_price),
        "best_bid_price_usd": option_price_to_usd(
            best_bid,
            underlying_price_float,
        ),
        "best_ask_price_usd": option_price_to_usd(
            best_ask,
            underlying_price_float,
        ),
        "mark_price_usd": option_price_to_usd(
            mark_price,
            underlying_price_float,
        ),
        "last_price_usd": option_price_to_usd(
            last_price,
            underlying_price_float,
        ),
        "bid_iv": normalize_deribit_iv(ticker.get("bid_iv")),
        "ask_iv": normalize_deribit_iv(ticker.get("ask_iv")),
        "mark_iv": normalize_deribit_iv(ticker.get("mark_iv")),
        "open_interest": safe_float(ticker.get("open_interest")),
        "state": str(ticker.get("state")),
        "delta": extract_greek(ticker, "delta"),
        "gamma": extract_greek(ticker, "gamma"),
        "theta": extract_greek(ticker, "theta"),
        "vega": extract_greek(ticker, "vega"),
        "rho": extract_greek(ticker, "rho"),
        "timestamp": str(ticker_timestamp) if ticker_timestamp is not None else None,
    }


def filter_options_near_atm(
    instruments: pd.DataFrame,
    underlying_price: float,
    min_days_to_expiry: int = 1,
    max_days_to_expiry: int = 45,
    strikes_each_side: int = 5,
) -> pd.DataFrame:
    """
    Filter options to a manageable near-ATM chain.

    This avoids making hundreds of ticker API calls.
    """

    if instruments.empty:
        return instruments

    if underlying_price <= 0:
        raise ValueError("underlying_price must be greater than 0.")

    if "expiration_datetime" not in instruments.columns:
        raise ValueError("instruments must contain expiration_datetime column.")

    if "strike" not in instruments.columns:
        raise ValueError("instruments must contain strike column.")

    if "option_type" not in instruments.columns:
        raise ValueError("instruments must contain option_type column.")

    data = instruments.copy()

    now_utc = pd.Timestamp.now(tz="UTC")

    expiration_series = pd.to_datetime(
        data["expiration_datetime"],
        utc=True,
        errors="coerce",
    )

    data["days_to_expiry"] = (
        expiration_series - now_utc
    ).dt.total_seconds() / 86400.0

    data = data[
        (data["days_to_expiry"] >= float(min_days_to_expiry))
        & (data["days_to_expiry"] <= float(max_days_to_expiry))
    ].copy()

    if data.empty:
        return data

    selected_frames: list[pd.DataFrame] = []

    for _, expiry_group in data.groupby("expiration_datetime"):
        group = expiry_group.copy()

        group["distance_to_atm"] = (
            pd.to_numeric(group["strike"], errors="coerce") - underlying_price
        ).abs()

        calls = group[group["option_type"] == "call"].copy()
        puts = group[group["option_type"] == "put"].copy()

        calls = calls.sort_values("distance_to_atm").head(strikes_each_side * 2 + 1)
        puts = puts.sort_values("distance_to_atm").head(strikes_each_side * 2 + 1)

        if not calls.empty:
            selected_frames.append(calls)

        if not puts.empty:
            selected_frames.append(puts)

    if not selected_frames:
        return pd.DataFrame()

    filtered = pd.concat(selected_frames, ignore_index=True)
    filtered = filtered.drop_duplicates(subset=["instrument_name"])
    filtered = filtered.sort_values(
        by=["expiration_datetime", "strike", "option_type"]
    ).reset_index(drop=True)

    return filtered


def build_live_eth_option_chain(
    config: DeribitConfig | None = None,
    min_days_to_expiry: int = 1,
    max_days_to_expiry: int = 45,
    strikes_each_side: int = 5,
    save_csv: bool = True,
) -> pd.DataFrame:
    """
    Build live ETH option chain from Deribit.

    Steps:
    1. fetch active ETH option instruments
    2. fetch ETH underlying price
    3. filter to near-ATM options
    4. fetch ticker for each selected option
    5. return clean DataFrame
    """

    if config is None:
        config = DeribitConfig()

    Path(config.output_folder).mkdir(parents=True, exist_ok=True)

    print("Fetching ETH option instruments from Deribit...")

    instruments = fetch_eth_option_instruments(
        config=config,
        expired=False,
    )

    if instruments.empty:
        raise RuntimeError("No ETH option instruments returned from Deribit.")

    print(f"Active ETH option instruments: {len(instruments)}")

    print("Fetching ETH underlying price from Deribit...")

    underlying_price = fetch_eth_underlying_price(config=config)

    print(f"ETH underlying price: ${underlying_price:,.2f}")

    filtered_instruments = filter_options_near_atm(
        instruments=instruments,
        underlying_price=underlying_price,
        min_days_to_expiry=min_days_to_expiry,
        max_days_to_expiry=max_days_to_expiry,
        strikes_each_side=strikes_each_side,
    )

    if filtered_instruments.empty:
        raise RuntimeError("No options matched the near-ATM filter.")

    print(f"Selected near-ATM options: {len(filtered_instruments)}")
    print("Fetching option tickers...")

    rows: list[dict[str, Any]] = []

    for _, instrument in filtered_instruments.iterrows():
        instrument_name_value = instrument.get("instrument_name")

        if instrument_name_value is None:
            continue

        instrument_name = str(instrument_name_value)

        try:
            ticker = fetch_deribit_ticker(
                instrument_name=instrument_name,
                config=config,
            )

            row = ticker_to_option_row(
                instrument=instrument,
                ticker=ticker,
                fallback_underlying_price=underlying_price,
            )

            rows.append(row)

        except Exception as error:
            print(f"Warning: failed ticker for {instrument_name}: {error}")

        time.sleep(config.request_sleep_seconds)

    option_chain = pd.DataFrame(rows)

    if option_chain.empty:
        raise RuntimeError("No option ticker rows were successfully fetched.")

    option_chain = option_chain.sort_values(
        by=["expiration_datetime", "strike", "option_type"],
        ascending=True,
    ).reset_index(drop=True)

    if save_csv:
        output_path = Path(config.output_folder) / "live_eth_option_chain.csv"
        option_chain.to_csv(output_path, index=False)
        print(f"Saved live option chain to: {output_path}")

    return option_chain


def print_option_chain_summary(option_chain: pd.DataFrame) -> None:
    """
    Print readable summary of the live option chain.
    """

    if option_chain.empty:
        print("Option chain is empty.")
        return

    print("\n========== LIVE ETH OPTION CHAIN SUMMARY ==========")
    print(f"Rows:                 {len(option_chain)}")

    if "underlying_price" in option_chain.columns:
        underlying_series = pd.to_numeric(
            option_chain["underlying_price"],
            errors="coerce",
        ).dropna()

        if not underlying_series.empty:
            underlying_price = float(underlying_series.iloc[0])
            print(f"Underlying ETH price: ${underlying_price:,.2f}")

    if "expiration_datetime" in option_chain.columns:
        print(f"Expiries:             {option_chain['expiration_datetime'].nunique()}")

    if "strike" in option_chain.columns:
        strike_series = pd.to_numeric(option_chain["strike"], errors="coerce").dropna()

        if not strike_series.empty:
            print(f"Min strike:           ${strike_series.min():,.2f}")
            print(f"Max strike:           ${strike_series.max():,.2f}")

    if "mark_iv" in option_chain.columns:
        clean_iv = pd.to_numeric(option_chain["mark_iv"], errors="coerce").dropna()

        if not clean_iv.empty:
            print(f"Median mark IV:       {clean_iv.median():.2%}")
            print(f"Min mark IV:          {clean_iv.min():.2%}")
            print(f"Max mark IV:          {clean_iv.max():.2%}")

    print("===================================================\n")

    display_columns = [
        "instrument_name",
        "option_type",
        "strike",
        "days_to_expiry",
        "underlying_price",
        "best_bid_price_usd",
        "best_ask_price_usd",
        "mark_price_usd",
        "mark_iv",
        "delta",
        "gamma",
        "vega",
    ]

    available_columns = [
        column for column in display_columns if column in option_chain.columns
    ]

    print(option_chain[available_columns].head(20).to_string(index=False))


if __name__ == "__main__":
    deribit_config = DeribitConfig(
        currency="ETH",
        kind="option",
        testnet=False,
        request_sleep_seconds=0.15,
        output_folder="outputs",
    )

    chain = build_live_eth_option_chain(
        config=deribit_config,
        min_days_to_expiry=1,
        max_days_to_expiry=45,
        strikes_each_side=4,
        save_csv=True,
    )

    print_option_chain_summary(chain)
