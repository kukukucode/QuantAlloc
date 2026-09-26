"""Tests for portfolio risk contribution."""

import pandas as pd
import pytest

from src.risk import portfolio_volatility, risk_contribution


def test_portfolio_volatility_for_uncorrelated_assets() -> None:
    weights = pd.Series({"A": 0.5, "B": 0.5})
    covariance = pd.DataFrame(
        [[0.04, 0.0], [0.0, 0.04]],
        index=["A", "B"],
        columns=["A", "B"],
    )

    assert portfolio_volatility(weights, covariance) == pytest.approx(
        0.04**0.5 / 2**0.5
    )


def test_equal_assets_make_equal_risk_contributions() -> None:
    weights = pd.Series({"A": 0.5, "B": 0.5})
    covariance = pd.DataFrame(
        [[0.04, 0.0], [0.0, 0.04]],
        index=["A", "B"],
        columns=["A", "B"],
    )

    result = risk_contribution(weights, covariance)

    assert result["risk_contribution_pct"].to_list() == pytest.approx(
        [0.5, 0.5]
    )
    assert result["risk_contribution"].sum() == pytest.approx(
        portfolio_volatility(weights, covariance)
    )


def test_risk_contribution_requires_matching_assets() -> None:
    weights = pd.Series({"A": 0.5, "C": 0.5})
    covariance = pd.DataFrame(
        [[0.04, 0.0], [0.0, 0.04]],
        index=["A", "B"],
        columns=["A", "B"],
    )

    with pytest.raises(ValueError, match="assets must match"):
        risk_contribution(weights, covariance)


@pytest.mark.parametrize("calculation", [portfolio_volatility, risk_contribution])
def test_risk_rejects_duplicate_weight_assets(calculation) -> None:
    weights = pd.Series([0.2, 0.5, 0.5], index=["A", "A", "B"])
    covariance = pd.DataFrame(
        [[0.04, 0.0], [0.0, 0.16]], index=["A", "B"], columns=["A", "B"]
    )
    with pytest.raises(ValueError, match="index must be unique"):
        calculation(weights, covariance)


def test_risk_rejects_duplicate_covariance_assets() -> None:
    covariance = pd.DataFrame(
        [[0.04, 0.0], [0.0, 0.16]], index=["A", "A"], columns=["A", "A"]
    )
    with pytest.raises(ValueError, match="assets must be unique"):
        portfolio_volatility(pd.Series({"A": 1.0}), covariance)
