"""Regression tests for benchmark date alignment."""

import numpy as np
import pandas as pd
import pytest

from src.benchmark import align_returns, compare_with_benchmark


def test_benchmark_comparison_trims_only_boundary_dates() -> None:
    dates = pd.bdate_range("2024-01-01", periods=5)
    portfolio = pd.Series([0.01, 0.02, -0.01], index=dates[1:4])
    benchmark = pd.Series([0.5, 0.01, 0.02, -0.01, -0.5], index=dates)

    result = compare_with_benchmark(portfolio, benchmark)

    cumulative = result["cumulative_returns"]
    assert cumulative.index.equals(dates[1:4])
    assert cumulative["benchmark"].iloc[-1] == pytest.approx(1.01 * 1.02 * 0.99 - 1)
    pd.testing.assert_series_equal(
        cumulative["portfolio"], cumulative["benchmark"], check_names=False
    )


@pytest.mark.parametrize("missing_input", ["portfolio", "benchmark"])
def test_benchmark_alignment_rejects_internal_missing_date(missing_input: str) -> None:
    dates = pd.bdate_range("2024-01-01", periods=3)
    complete = pd.Series([0.01, 0.10, 0.02], index=dates)
    incomplete = complete.drop(dates[1])
    portfolio, benchmark = (
        (incomplete, complete) if missing_input == "portfolio" else (complete, incomplete)
    )

    with pytest.raises(ValueError, match="missing dates"):
        align_returns(portfolio, benchmark)


def test_benchmark_alignment_rejects_nan_instead_of_dropping_day() -> None:
    dates = pd.bdate_range("2024-01-01", periods=3)
    portfolio = pd.Series([0.01, np.nan, 0.02], index=dates)
    benchmark = pd.Series([0.01, 0.10, 0.02], index=dates)

    with pytest.raises(ValueError, match="must not contain NaN"):
        align_returns(portfolio, benchmark)


def test_benchmark_alignment_sorts_dates_without_changing_returns() -> None:
    dates = pd.bdate_range("2024-01-01", periods=3)
    returns = pd.Series([0.01, 0.10, 0.02], index=dates)

    portfolio, benchmark = align_returns(returns.iloc[::-1], returns)

    pd.testing.assert_series_equal(portfolio, returns.rename("portfolio"))
    pd.testing.assert_series_equal(benchmark, returns.rename("benchmark"))
