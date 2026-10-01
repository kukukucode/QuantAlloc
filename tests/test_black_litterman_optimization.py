"""Integration tests for Black-Litterman maximum-Sharpe allocation."""

import numpy as np
import pandas as pd
import pytest

from src.black_litterman import implied_equilibrium_returns
from src.optimization import black_litterman_weights


@pytest.fixture
def allocation_inputs() -> tuple[
    pd.Series, pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame
]:
    covariance = pd.DataFrame(
        np.diag([0.04, 0.04]), index=["A", "B"], columns=["A", "B"]
    )
    prior = implied_equilibrium_returns(
        pd.Series({"A": 0.5, "B": 0.5}), covariance, risk_aversion=5.0
    )
    view_matrix = pd.DataFrame(
        np.eye(2), index=["view_A", "view_B"], columns=["A", "B"]
    )
    view_returns = pd.Series({"view_A": 0.2, "view_B": 0.1})
    view_uncertainty = pd.DataFrame(
        np.diag([0.002, 0.002]),
        index=view_matrix.index,
        columns=view_matrix.index,
    )
    return prior, covariance, view_matrix, view_returns, view_uncertainty


@pytest.mark.parametrize(
    ("tau", "risk_free_rate", "expected_weight_a"),
    [(0.05, 0.0, 0.6), (0.1, 0.0, 0.625), (0.05, 0.05, 2.0 / 3.0)],
)
def test_bl_allocation_matches_analytical_weights(
    allocation_inputs: tuple,
    tau: float,
    risk_free_rate: float,
    expected_weight_a: float,
) -> None:
    # Independent assets: posterior_i is the precision-weighted prior/view
    # mean, and optimal weights are proportional to excess means / variance.
    result = black_litterman_weights(
        *allocation_inputs, tau=tau, risk_free_rate=risk_free_rate
    )

    expected = pd.Series(
        {"A": expected_weight_a, "B": 1.0 - expected_weight_a},
        name="black_litterman",
    )
    pd.testing.assert_series_equal(result, expected, atol=1e-6, rtol=0.0)
    assert result.sum() == pytest.approx(1.0)
    assert (result >= 0.0).all()
    assert (result <= 1.0).all()


def test_bl_allocation_aligns_assets_and_views(allocation_inputs: tuple) -> None:
    prior, covariance, p, q, omega = allocation_inputs
    result = black_litterman_weights(
        prior,
        covariance.loc[["B", "A"], ["B", "A"]],
        p.loc[["view_B", "view_A"]],
        q,
        omega,
    )

    expected = pd.Series({"B": 0.4, "A": 0.6}, name="black_litterman")
    pd.testing.assert_series_equal(result, expected, atol=1e-6, rtol=0.0)


def test_bl_allocation_responds_to_views(allocation_inputs: tuple) -> None:
    prior, covariance, p, q, omega = allocation_inputs
    neutral = black_litterman_weights(
        prior, covariance, p, pd.Series(0.1, index=q.index), omega
    )
    favors_a = black_litterman_weights(*allocation_inputs)
    favors_b = black_litterman_weights(
        prior, covariance, p, pd.Series({"view_A": 0.1, "view_B": 0.2}), omega
    )

    assert neutral.to_list() == pytest.approx([0.5, 0.5], abs=1e-6)
    assert favors_a["A"] > neutral["A"] > favors_b["A"]
    assert favors_b.to_list() == pytest.approx([0.4, 0.6], abs=1e-6)


def test_bl_allocation_supports_relative_views(allocation_inputs: tuple) -> None:
    prior, covariance, _, _, _ = allocation_inputs
    p = pd.DataFrame([[1.0, -1.0]], index=["relative"], columns=["A", "B"])
    q = pd.Series({"relative": 0.1})
    omega = pd.DataFrame([[0.004]], index=q.index, columns=q.index)

    result = black_litterman_weights(prior, covariance, p, q, omega)

    # posterior = [0.125, 0.075], so the equal-variance optimum is 62.5/37.5.
    assert result.to_list() == pytest.approx([0.625, 0.375], abs=1e-6)


def test_bl_allocation_respects_long_only_with_negative_view(
    allocation_inputs: tuple,
) -> None:
    prior, covariance, p, _, omega = allocation_inputs
    q = pd.Series({"view_A": -1.0, "view_B": 0.1})

    result = black_litterman_weights(prior, covariance, p, q, omega)

    assert result.to_list() == pytest.approx([0.0, 1.0], abs=1e-6)


def test_bl_allocation_rejects_asset_mismatch(allocation_inputs: tuple) -> None:
    prior, covariance, p, q, omega = allocation_inputs

    with pytest.raises(
        ValueError, match="equilibrium_returns and covariance assets must match"
    ):
        black_litterman_weights(
            prior.rename(index={"A": "C"}), covariance, p, q, omega
        )


def test_bl_allocation_rejects_non_finite_risk_free_rate(
    allocation_inputs: tuple,
) -> None:
    with pytest.raises(ValueError, match="risk_free_rate must be finite"):
        black_litterman_weights(*allocation_inputs, risk_free_rate=np.nan)
