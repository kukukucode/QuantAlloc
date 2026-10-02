"""Tests for classical portfolio optimization."""

import numpy as np
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


@pytest.mark.parametrize("risk_free_rate", [0.0, 0.05])
def test_maximum_sharpe_with_negative_excess_returns_chooses_best_asset(
    risk_free_rate: float,
) -> None:
    covariance = pd.DataFrame(
        [[0.04, -0.06], [-0.06, 0.10]],
        index=["A", "B"],
        columns=["A", "B"],
    )
    returns = pd.Series({"A": -0.02 + risk_free_rate, "B": -0.09 + risk_free_rate})

    result = maximum_sharpe_weights(returns, covariance, risk_free_rate)

    assert result.to_list() == pytest.approx([1.0, 0.0])
    performance = portfolio_performance(result, returns, covariance, risk_free_rate)
    assert performance["sharpe"] == pytest.approx(-0.1)


def test_maximum_sharpe_accepts_zero_excess_return(covariance: pd.DataFrame) -> None:
    result = maximum_sharpe_weights(
        pd.Series({"A": 0.03, "B": 0.01}), covariance, risk_free_rate=0.03
    )

    assert result.to_list() == pytest.approx([1.0, 0.0])


@pytest.mark.parametrize("asset_count", [1, 2])
def test_efficient_frontier_with_equal_means_contains_only_minimum_risk(
    asset_count: int,
) -> None:
    assets = ["A", "B"][:asset_count]
    covariance = pd.DataFrame(
        np.diag([0.04, 0.16][:asset_count]), index=assets, columns=assets
    )
    returns = pd.Series(0.1, index=assets)

    result = efficient_frontier(returns, covariance, points=3)

    assert len(result) == 3
    assert result["return"].to_list() == pytest.approx([0.1] * 3)
    variance = 0.04 if asset_count == 1 else 0.032
    assert result["volatility"].to_list() == pytest.approx([variance**0.5] * 3)


@pytest.mark.parametrize("scale", [1e-9, 1e-6, 1.0, 1e6])
def test_minimum_variance_is_invariant_to_covariance_scale(scale: float) -> None:
    covariance = pd.DataFrame(
        np.diag([0.04, 0.16]), index=["A", "B"], columns=["A", "B"]
    )

    result = minimum_variance_weights(covariance * scale)

    assert result.to_list() == pytest.approx([0.8, 0.2], abs=1e-6)


def test_frontier_preserves_targets_when_covariance_is_small() -> None:
    covariance = pd.DataFrame(
        np.diag([0.04, 0.16]) * 1e-6, index=["A", "B"], columns=["A", "B"]
    )
    returns = pd.Series({"A": 0.05, "B": 0.20})

    result = efficient_frontier(returns, covariance, points=5)

    targets = np.linspace(0.08, 0.20, 5)
    weight_b = (targets - 0.05) / 0.15
    expected_volatility = np.sqrt(
        ((1.0 - weight_b) ** 2 * 0.04 + weight_b**2 * 0.16) * 1e-6
    )
    assert result["return"].to_numpy() == pytest.approx(targets, abs=1e-8)
    assert result["volatility"].to_numpy() == pytest.approx(expected_volatility, rel=1e-5)


def test_optimization_rejects_duplicate_expected_return_assets(
    covariance: pd.DataFrame,
) -> None:
    returns = pd.Series([0.1, 0.2, 0.05], index=["A", "A", "B"])
    with pytest.raises(ValueError, match="index must be unique"):
        maximum_sharpe_weights(returns, covariance)


def test_performance_rejects_duplicate_weights(covariance: pd.DataFrame) -> None:
    weights = pd.Series([0.2, 0.5, 0.5], index=["A", "A", "B"])
    with pytest.raises(ValueError, match="index must be unique"):
        portfolio_performance(weights, pd.Series({"A": 0.1, "B": 0.05}), covariance)


def test_frontier_endpoint_with_tied_maximum_returns_minimizes_risk() -> None:
    assets = ["A", "B", "C"]
    covariance = pd.DataFrame(np.diag([0.09, 0.04, 0.16]), index=assets, columns=assets)
    returns = pd.Series({"A": 0.01, "B": 0.20, "C": 0.20})

    result = efficient_frontier(returns, covariance, points=5)

    assert result.iloc[-1]["return"] == pytest.approx(0.20)
    assert result.iloc[-1]["volatility"] == pytest.approx(np.sqrt(0.032))


def test_frontier_handles_ten_asset_maximum_return_endpoint() -> None:
    generator = np.random.default_rng(19)
    daily = generator.normal(0.0005, 0.01, size=(757, 11))[:, :10]
    prices = pd.DataFrame(100.0 * np.cumprod(1.0 + daily, axis=0))
    asset_returns = prices.pct_change(fill_method=None).dropna()
    covariance = asset_returns.cov() * 252
    means = expected_returns(asset_returns)

    result = efficient_frontier(means, covariance, points=10)

    highest_mean_asset = means.idxmax()
    assert result.iloc[-1]["return"] == pytest.approx(means.max())
    assert result.iloc[-1]["volatility"] == pytest.approx(
        np.sqrt(covariance.loc[highest_mean_asset, highest_mean_asset])
    )
