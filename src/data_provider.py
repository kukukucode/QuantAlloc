"""Market data retrieval for QuantAlloc."""

import pandas as pd
import yfinance as yf


def fetch_prices(
    tickers: list[str],
    start: str,
    end: str | None = None,
) -> pd.DataFrame:
    """Fetch adjusted closing prices from Yahoo Finance.

    Args:
        tickers: Yahoo Finance ticker symbols.
        start: Inclusive start date in YYYY-MM-DD format.
        end: Exclusive end date in YYYY-MM-DD format. Defaults to today.

    Returns:
        A date-indexed DataFrame with one adjusted-price column per ticker.

    Raises:
        ValueError: If the input is invalid or no usable prices are returned.
    """
    if not tickers:
        raise ValueError("tickers must not be empty")

    normalized_tickers = [ticker.strip() for ticker in tickers]
    if any(not ticker for ticker in normalized_tickers):
        raise ValueError("tickers must contain non-empty strings")
    if len(set(normalized_tickers)) != len(normalized_tickers):
        raise ValueError("tickers must not contain duplicates")

    raw_data = yf.download(
        normalized_tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if raw_data.empty:
        raise ValueError("no price data was returned")

    try:
        close = raw_data["Close"]
    except KeyError as exc:
        raise ValueError("downloaded data does not contain closing prices") from exc

    if isinstance(close, pd.Series):
        prices = close.to_frame(name=normalized_tickers[0])
    else:
        prices = close.copy()

    missing_tickers = [
        ticker for ticker in normalized_tickers if ticker not in prices.columns
    ]
    if missing_tickers:
        missing = ", ".join(missing_tickers)
        raise ValueError(f"no closing prices were returned for: {missing}")

    prices = prices.loc[:, normalized_tickers]
    prices.index = pd.to_datetime(prices.index)
    prices = prices[~prices.index.duplicated(keep="last")].sort_index()
    prices = prices.apply(pd.to_numeric, errors="coerce").ffill().dropna(how="any")

    if prices.empty:
        raise ValueError("no complete price observations were returned")

    prices.index.name = "Date"
    return prices
