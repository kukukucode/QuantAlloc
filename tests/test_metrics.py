"""Tests for portfolio performance metrics."""

import math

import pandas as pd
import pytest

from src.metrics import (
    annualized_volatility,
    cagr,
    max_drawdown,
    sharpe_ratio,
    wealth_index,
)


def test_wealth_index_compounds_returns() -> None:
    returns = pd.Series([0.10, -0.10], name="portfolio_return")

    result = wealth_index(returns)

    expected = pd.Series([1.10, 0.99], name="wealth")
    pd.testing.assert_series_equal(result, expected)


def test_cagr_uses_actual_date_span() -> None:
    returns = pd.Series(
        [0.0, 0.10],
        index=pd.to_datetime(["2024-01-01", "2024-12-31"]),
    )
    expected = 1.10 ** (365.25 / 365.0) - 1.0

    assert cagr(returns) == pytest.approx(expected)


def test_annualized_volatility_is_zero_for_constant_returns() -> None:
    returns = pd.Series([0.01, 0.01, 0.01, 0.01])

    assert annualized_volatility(returns) == pytest.approx(0.0)


def test_sharpe_ratio() -> None:
    returns = pd.Series([0.01, -0.01, 0.02, -0.005])
    expected = (
        returns.mean() * 252
        / (returns.std(ddof=1) * math.sqrt(252))
    )

    assert sharpe_ratio(returns) == pytest.approx(expected)


def test_sharpe_ratio_is_nan_when_volatility_is_zero() -> None:
    returns = pd.Series([0.01, 0.01, 0.01, 0.01])

    assert math.isnan(sharpe_ratio(returns))


def test_max_drawdown() -> None:
    returns = pd.Series([0.20, -0.25])

    assert max_drawdown(returns) == pytest.approx(-0.25)
