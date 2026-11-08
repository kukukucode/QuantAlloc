"""Regression tests for hierarchical risk parity on historical windows."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import walk_forward_backtest
from src.hrp import hierarchical_risk_parity_weights


def test_hrp_single_asset() -> None:
    covariance = pd.DataFrame([[0.04]], index=["A"], columns=["A"])
    result = hierarchical_risk_parity_weights(covariance)
    assert result.name == "hierarchical_risk_parity"
    assert result.loc["A"] == 1.0


def test_hrp_two_assets_equal_correlation_inverse_variance() -> None:
    covariance = pd.DataFrame(
        [[0.04, 0.0], [0.0, 0.16]], index=["A", "B"], columns=["A", "B"]
    )
    result = hierarchical_risk_parity_weights(covariance)
    np.testing.assert_allclose(result.to_numpy(), [0.8, 0.2], atol=1e-12)


def test_hrp_identical_independent_assets_equal_weight() -> None:
    assets = list("ABCD")
    covariance = pd.DataFrame(np.eye(4) * 0.04, index=assets, columns=assets)
    result = hierarchical_risk_parity_weights(covariance)
    np.testing.assert_allclose(result.to_numpy(), np.full(4, 0.25), atol=1e-12)


def test_hrp_weights_are_aligned_long_only_and_fully_invested() -> None:
    assets = ["Toyota", "Sony", "MUFG", "NTT", "Nintendo"]
    factor = np.array([0.6, 0.4, -0.1, 0.2, 0.3])
    matrix = 0.02 * np.outer(factor, factor) + np.diag(
        [0.03, 0.05, 0.01, 0.06, 0.08]
    )
    covariance = pd.DataFrame(matrix, index=assets, columns=assets)
    weights = hierarchical_risk_parity_weights(covariance)
    assert weights.index.tolist() == assets
    assert (weights >= 0.0).all()
    assert (weights <= 1.0).all()
    assert weights.sum() == pytest.approx(1.0)


@pytest.mark.parametrize(
    "covariance",
    [
        pd.DataFrame(),
        pd.DataFrame(
            [[0.04, 0.02], [0.01, 0.09]],
            index=["A", "B"],
            columns=["A", "B"],
        ),
        pd.DataFrame(
            [[0.0, 0.0], [0.0, 0.09]],
            index=["A", "B"],
            columns=["A", "B"],
        ),
        pd.DataFrame(
            [[0.04, 0.06], [0.06, 0.04]],
            index=["A", "B"],
            columns=["A", "B"],
        ),
    ],
)
def test_hrp_rejects_invalid_covariance(covariance: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        hierarchical_risk_parity_weights(covariance)


def test_hrp_walk_forward_uses_only_past_returns() -> None:
    rng = np.random.default_rng(33)
    dates = pd.bdate_range("2022-01-03", periods=200)
    returns = pd.DataFrame(
        rng.normal(0.0005, 0.015, (200, 5)),
        index=dates,
        columns=list("ABCDE"),
    )
    baseline = walk_forward_backtest(
        returns,
        "hierarchical_risk_parity",
        estimation_window=100,
        holding_period=20,
    )
    altered = returns.copy()
    altered.iloc[100:] *= 1.2
    changed = walk_forward_backtest(
        altered,
        "hierarchical_risk_parity",
        estimation_window=100,
        holding_period=20,
    )
    np.testing.assert_allclose(
        baseline.target_weights.iloc[0],
        changed.target_weights.iloc[0],
        rtol=0.0,
        atol=0.0,
    )
    assert (baseline.target_weights >= 0.0).all().all()
    np.testing.assert_allclose(baseline.target_weights.sum(axis=1), 1.0)


def test_hrp_single_strategy_dashboard(sample_market_prices) -> None:
    from src.dashboard import run_sample_analysis

    result = run_sample_analysis(
        *sample_market_prices, strategy="hierarchical_risk_parity"
    )
    assert result.strategy == "hierarchical_risk_parity"
    assert "Hierarchical Risk Parity (HRP)" in result.performance.index
    assert result.backtest.net_returns.notna().all()
