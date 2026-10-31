"""Tests for historical VaR and CVaR."""

import numpy as np
import pandas as pd
import pytest

from src.risk import conditional_value_at_risk, value_at_risk


def test_value_at_risk() -> None:
    returns = pd.Series([-0.10, -0.04, 0.01, 0.02])

    assert value_at_risk(returns, 0.5) == pytest.approx(0.015)


def test_conditional_value_at_risk() -> None:
    returns = pd.Series([-0.10, -0.04, 0.01, 0.02])

    assert conditional_value_at_risk(returns, 0.5) == pytest.approx(0.07)


def test_cvar_fractional_tail() -> None:
    returns = pd.Series([-0.20, 0.0, 0.10])

    expected = 0.20 / 1.5
    assert conditional_value_at_risk(returns, 0.5) == pytest.approx(expected)


def test_cvar_not_below_var() -> None:
    returns = pd.Series([-0.20, -0.05, 0.01, 0.02, 0.03])

    assert conditional_value_at_risk(returns) >= value_at_risk(returns)


def test_no_losses_returns_zero() -> None:
    returns = pd.Series([0.01, 0.02, 0.03])

    assert value_at_risk(returns) == 0.0
    assert conditional_value_at_risk(returns) == 0.0


@pytest.mark.parametrize(
    "level",
    [0.0, 1.0, -0.1, 1.1, True, np.nan, np.inf, "0.95"],
)
@pytest.mark.parametrize(
    "metric",
    [value_at_risk, conditional_value_at_risk],
)
def test_invalid_confidence_level(metric, level) -> None:
    with pytest.raises(ValueError, match="confidence_level"):
        metric(pd.Series([0.01, -0.02]), level)


@pytest.mark.parametrize(
    "returns",
    [
        pd.Series(dtype=float),
        pd.Series([0.01, np.nan]),
        pd.Series(["invalid", "data"]),
        pd.Series([-1.2, 0.1]),
    ],
)
@pytest.mark.parametrize(
    "metric",
    [value_at_risk, conditional_value_at_risk],
)
def test_invalid_returns(metric, returns) -> None:
    with pytest.raises(ValueError):
        metric(returns)
