"""Deterministic offline prices for the dashboard and app tests."""

import numpy as np
import pandas as pd
import pytest

from src.sample_config import BENCHMARK_TICKER, TICKERS


@pytest.fixture
def sample_market_prices() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2021-01-01", periods=694)
    symbols = TICKERS + [BENCHMARK_TICKER]
    generator = np.random.default_rng(31)
    returns = generator.normal(0.0005, 0.012, (len(dates), len(symbols)))
    prices = pd.DataFrame(
        100.0 * (1.0 + returns).cumprod(axis=0), index=dates, columns=symbols
    )
    return prices[TICKERS].copy(), prices[[BENCHMARK_TICKER]].copy()
