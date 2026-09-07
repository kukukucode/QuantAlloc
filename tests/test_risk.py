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
