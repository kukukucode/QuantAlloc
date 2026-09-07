"""Tests for classical portfolio optimization."""

import pandas as pd
import pytest

from src.optimization import (
    efficient_frontier,
    expected_returns,
    maximum_sharpe_weights,
    minimum_variance_weights,
    portfolio_performance,
)


@pytest.fixture
def covariance() -> pd.DataFrame:
    return pd.DataFrame(
        [[0.04, 0.0], [0.0, 0.04]],
        index=["A", "B"],
        columns=["A", "B"],
    )


def test_expected_returns_annualizes_historical_mean() -> None:
    returns = pd.DataFrame({"A": [0.01, 0.03], "B": [0.00, 0.02]})

    result = expected_returns(returns)

    expected = pd.Series(
        {"A": 0.02 * 252, "B": 0.01 * 252},
        name="expected_return",
    )
    pd.testing.assert_series_equal(result, expected)


def test_minimum_variance_is_equal_for_identical_uncorrelated_assets(
    covariance: pd.DataFrame,
) -> None:
    result = minimum_variance_weights(covariance)

    assert result.to_list() == pytest.approx([0.5, 0.5])


def test_optimized_weights_follow_constraints(
    covariance: pd.DataFrame,
) -> None:
    returns = pd.Series({"A": 0.20, "B": 0.05})

    for weights in (
        minimum_variance_weights(covariance),
        maximum_sharpe_weights(returns, covariance),
    ):
        assert weights.sum() == pytest.approx(1.0)
        assert (weights >= 0.0).all()
        assert (weights <= 1.0).all()


def test_maximum_sharpe_favors_higher_risk_adjusted_return(
    covariance: pd.DataFrame,
) -> None:
    returns = pd.Series({"A": 0.20, "B": 0.05})

    result = maximum_sharpe_weights(returns, covariance)

    assert result["A"] > result["B"]
    assert result["A"] == pytest.approx(0.8, abs=1e-4)


def test_portfolio_performance(covariance: pd.DataFrame) -> None:
    weights = pd.Series({"A": 0.5, "B": 0.5})
    returns = pd.Series({"A": 0.20, "B": 0.10})

    result = portfolio_performance(weights, returns, covariance)

    assert result["return"] == pytest.approx(0.15)
    assert result["volatility"] == pytest.approx((0.02) ** 0.5)
    assert result["sharpe"] == pytest.approx(0.15 / (0.02**0.5))


def test_efficient_frontier_has_requested_points(
    covariance: pd.DataFrame,
) -> None:
    returns = pd.Series({"A": 0.20, "B": 0.05})

    result = efficient_frontier(returns, covariance, points=5)

    assert list(result.columns) == ["return", "volatility"]
    assert len(result) == 5
    assert result["return"].is_monotonic_increasing
    assert result["volatility"].is_monotonic_increasing
