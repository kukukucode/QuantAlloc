"""Tests for bootstrap resampling."""

import numpy as np
import pandas as pd
import pytest
from src.portfolio import calculate_portfolio_returns
from src.evaluation import performance_summary

from src.bootstrap import (
    bootstrap_portfolio_summary,
    bootstrap_metric_intervals,
    bootstrap_indices,
    resample_asset_returns,
    bootstrap_portfolio_metrics,
    bootstrap_metric_distribution,
)


def test_bootstrap_indices_reproducible() -> None:
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    result1 = bootstrap_indices(100, rng1)
    result2 = bootstrap_indices(100, rng2)

    np.testing.assert_array_equal(result1, result2)


def test_bootstrap_indices_valid_range() -> None:
    rng = np.random.default_rng(123)

    indices = bootstrap_indices(100, rng)

    assert len(indices) == 100
    assert np.issubdtype(indices.dtype, np.integer)
    assert np.all(indices >= 0)
    assert np.all(indices < 100)


def test_bootstrap_indices_allows_duplicates() -> None:
    rng = np.random.default_rng(42)

    indices = bootstrap_indices(100, rng)

    assert len(np.unique(indices)) < len(indices)


@pytest.mark.parametrize(
    "n_observations",
    [0, -1, 1.5, True],
)
def test_bootstrap_indices_rejects_invalid_size(
    n_observations,
) -> None:
    rng = np.random.default_rng(42)

    with pytest.raises(ValueError):
        bootstrap_indices(n_observations, rng)


def test_resample_asset_returns_reproducible() -> None:
    returns = pd.DataFrame(
        {
            "Toyota": [0.01, -0.02, 0.03],
            "Sony": [0.02, 0.01, -0.01],
        },
        index=pd.bdate_range("2024-01-01", periods=3),
    )

    result1 = resample_asset_returns(
        returns,
        np.random.default_rng(42),
    )
    result2 = resample_asset_returns(
        returns,
        np.random.default_rng(42),
    )

    pd.testing.assert_frame_equal(result1, result2)


def test_resample_asset_returns_preserves_complete_rows() -> None:
    returns = pd.DataFrame(
        {
            "Toyota": [0.01, -0.02, 0.03],
            "Sony": [0.02, 0.01, -0.01],
        },
        index=pd.bdate_range("2024-01-01", periods=3),
    )

    seed = 123
    expected_indices = bootstrap_indices(
        len(returns),
        np.random.default_rng(seed),
    )

    result = resample_asset_returns(
        returns,
        np.random.default_rng(seed),
    )

    expected = returns.iloc[expected_indices].copy()
    expected.index = returns.index

    pd.testing.assert_frame_equal(result, expected)


def test_bootstrap_portfolio_metrics() -> None:
    returns = pd.DataFrame(
        {
            "Toyota": [0.01, -0.02, 0.03, 0.01, -0.01, 0.02],
            "Sony": [0.02, 0.01, -0.01, 0.03, -0.02, 0.01],
        },
        index=pd.bdate_range("2024-01-01", periods=6),
    )

    weights = pd.Series(
        {
            "Toyota": 0.6,
            "Sony": 0.4,
        }
    )

    seed = 42

    result = bootstrap_portfolio_metrics(
        returns,
        weights,
        np.random.default_rng(seed),
        risk_free_rate=0.02,
    )

    indices = bootstrap_indices(
        len(returns),
        np.random.default_rng(seed),
    )

    sampled = returns.iloc[indices].copy()
    sampled.index = returns.index

    expected = performance_summary(
        calculate_portfolio_returns(sampled, weights),
        risk_free_rate=0.02,
    )

    assert result == pytest.approx(expected)
    assert set(result) == {
        "cagr",
        "volatility",
        "sharpe",
        "max_drawdown",
    }


def test_bootstrap_metric_distribution() -> None:
    returns = pd.DataFrame(
        {
            "Toyota": [0.01, -0.02, 0.03, 0.01, -0.01, 0.02],
            "Sony": [0.02, 0.01, -0.01, 0.03, -0.02, 0.01],
        },
        index=pd.bdate_range("2024-01-01", periods=6),
    )

    weights = pd.Series(
        {
            "Toyota": 0.6,
            "Sony": 0.4,
        }
    )

    result1 = bootstrap_metric_distribution(
        returns,
        weights,
        n_bootstrap=10,
        seed=42,
    )

    result2 = bootstrap_metric_distribution(
        returns,
        weights,
        n_bootstrap=10,
        seed=42,
    )

    assert len(result1) == 10
    assert set(result1.columns) == {
        "cagr",
        "volatility",
        "sharpe",
        "max_drawdown",
    }

    pd.testing.assert_frame_equal(result1, result2)


def test_bootstrap_metric_intervals() -> None:
    distribution = pd.DataFrame(
        {
            "sharpe": [0.0, 1.0, 2.0],
            "cagr": [0.1, 0.2, 0.3],
        }
    )

    result = bootstrap_metric_intervals(distribution)

    assert result.columns.tolist() == [
        "lower_2_5",
        "median",
        "upper_97_5",
    ]

    assert result.loc["sharpe"].to_numpy() == pytest.approx([0.05, 1.0, 1.95])
    assert result.loc["cagr"].to_numpy() == pytest.approx([0.105, 0.2, 0.295])


def test_bootstrap_portfolio_summary() -> None:
    returns = pd.DataFrame(
        {
            "Toyota": [0.01, -0.02, 0.03, 0.01, -0.01, 0.02],
            "Sony": [0.02, 0.01, -0.01, 0.03, -0.02, 0.01],
        },
        index=pd.bdate_range("2024-01-01", periods=6),
    )
    weights = pd.Series({"Toyota": 0.6, "Sony": 0.4})

    result = bootstrap_portfolio_summary(
        returns,
        weights,
        n_bootstrap=20,
        seed=42,
    )

    observed = performance_summary(calculate_portfolio_returns(returns, weights))

    assert result.columns.tolist() == [
        "observed",
        "lower_2_5",
        "median",
        "upper_97_5",
    ]

    assert result.index.tolist() == [
        "cagr",
        "volatility",
        "sharpe",
        "max_drawdown",
    ]

    assert result["observed"].to_dict() == pytest.approx(observed)

    expected_intervals = bootstrap_metric_intervals(
        bootstrap_metric_distribution(
            returns,
            weights,
            n_bootstrap=20,
            seed=42,
        )
    )

    pd.testing.assert_frame_equal(
        result.drop(columns="observed"),
        expected_intervals,
    )
