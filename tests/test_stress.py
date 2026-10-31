"""Tests for retrospective stress analysis."""

import numpy as np
import pandas as pd
import pytest

from src.stress import worst_benchmark_stress


@pytest.fixture
def returns():
    dates = pd.bdate_range("2024-01-01", periods=7)

    return pd.DataFrame(
        {
            "Strategy": [-0.05, 0.10, 0.0, -0.10, -0.10, 0.10, 0.50],
            "Safe": [0.01] * 7,
            "Never": [-0.01] * 7,
            "TOPIX": [-0.10, -0.10, 0.20, -0.20, -0.20, 0.10, 0.50],
        },
        index=dates,
    )


def test_selects_worst_benchmark_window(returns) -> None:
    result = worst_benchmark_stress(returns, window=2)

    assert result.start == returns.index[3]
    assert result.end == returns.index[4]
    assert result.summary.loc["TOPIX", "window_return"] == pytest.approx(-0.36)
    assert result.summary.loc["Strategy", "window_return"] == pytest.approx(-0.19)
    assert result.summary.loc["TOPIX", "max_drawdown"] == pytest.approx(-0.36)


def test_recovery_days_and_unrecovered(returns) -> None:
    result = worst_benchmark_stress(returns, window=2)

    assert result.summary.loc["Strategy", "recovery_days"] == 2
    assert result.summary.loc["TOPIX", "recovery_days"] == 2
    assert result.summary.loc["Safe", "recovery_days"] == 0
    assert pd.isna(result.summary.loc["Never", "recovery_days"])


def test_paths_include_first_stress_day(returns) -> None:
    result = worst_benchmark_stress(returns, window=2)

    assert result.cumulative_returns.index.equals(returns.index[3:5])
    assert result.cumulative_returns.loc[returns.index[3], "TOPIX"] == pytest.approx(
        -0.20
    )
    assert result.cumulative_returns.loc[returns.index[4], "TOPIX"] == pytest.approx(
        -0.36
    )


def test_does_not_modify_input(returns) -> None:
    original = returns.copy(deep=True)

    worst_benchmark_stress(returns, window=2)

    pd.testing.assert_frame_equal(returns, original)


@pytest.mark.parametrize("window", [0, 1, -1, 2.5, True, 100])
def test_rejects_invalid_window(returns, window) -> None:
    with pytest.raises(ValueError):
        worst_benchmark_stress(returns, window=window)


def test_rejects_missing_benchmark(returns) -> None:
    with pytest.raises(ValueError, match="benchmark"):
        worst_benchmark_stress(returns.drop(columns="TOPIX"), window=2)


def test_rejects_missing_values(returns) -> None:
    returns.iloc[0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        worst_benchmark_stress(returns, window=2)


def test_rejects_non_datetime_index(returns) -> None:
    returns.index = range(len(returns))

    with pytest.raises(TypeError, match="DatetimeIndex"):
        worst_benchmark_stress(returns, window=2)
