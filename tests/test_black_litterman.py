"""Tests for core Black-Litterman calculations."""

import numpy as np
import pandas as pd
import pytest

from src.black_litterman import implied_equilibrium_returns


@pytest.fixture
def covariance() -> pd.DataFrame:
    return pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.09]],
        index=["A", "B"],
        columns=["A", "B"],
    )


def test_implied_equilibrium_returns_match_formula(
    covariance: pd.DataFrame,
) -> None:
    market_weights = pd.Series({"A": 0.60, "B": 0.40})

    result = implied_equilibrium_returns(
        market_weights,
        covariance,
        risk_aversion=2.5,
    )

    expected = pd.Series(
        {"A": 0.07, "B": 0.105},
        name="implied_equilibrium_return",
    )
    pd.testing.assert_series_equal(result, expected)


def test_market_weights_are_aligned_to_covariance_assets(
    covariance: pd.DataFrame,
) -> None:
    market_weights = pd.Series({"B": 0.40, "A": 0.60})

    result = implied_equilibrium_returns(
        market_weights,
        covariance,
        risk_aversion=2.5,
    )

    assert result.index.to_list() == ["A", "B"]
    assert result.to_list() == pytest.approx([0.07, 0.105])


def test_risk_aversion_scales_implied_returns(
    covariance: pd.DataFrame,
) -> None:
    market_weights = pd.Series({"A": 0.60, "B": 0.40})

    low = implied_equilibrium_returns(
        market_weights,
        covariance,
        risk_aversion=1.0,
    )
    high = implied_equilibrium_returns(
        market_weights,
        covariance,
        risk_aversion=3.0,
    )

    pd.testing.assert_series_equal(high, low * 3.0)


@pytest.mark.parametrize(
    "risk_aversion",
    [0.0, -1.0, np.nan, np.inf, True, "invalid"],
)
def test_invalid_risk_aversion_is_rejected(
    covariance: pd.DataFrame,
    risk_aversion: object,
) -> None:
    market_weights = pd.Series({"A": 0.60, "B": 0.40})

    with pytest.raises(ValueError, match="positive finite number"):
        implied_equilibrium_returns(
            market_weights,
            covariance,
            risk_aversion=risk_aversion,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "market_weights",
    [
        pd.Series({"A": 0.60, "C": 0.40}),
        pd.Series({"A": 0.60, "B": 0.30}),
        pd.Series({"A": 1.10, "B": -0.10}),
        pd.Series({"A": np.nan, "B": np.nan}),
    ],
)
def test_invalid_market_weights_are_rejected(
    covariance: pd.DataFrame,
    market_weights: pd.Series,
) -> None:
    with pytest.raises(ValueError):
        implied_equilibrium_returns(
            market_weights,
            covariance,
            risk_aversion=2.5,
        )


def test_duplicate_market_weight_assets_are_rejected(
    covariance: pd.DataFrame,
) -> None:
    market_weights = pd.Series(
        [0.50, 0.25, 0.25],
        index=["A", "A", "B"],
    )

    with pytest.raises(ValueError, match="index must be unique"):
        implied_equilibrium_returns(
            market_weights,
            covariance,
            risk_aversion=2.5,
        )


@pytest.mark.parametrize("market_weights", [pd.Series(dtype=float), [0.5, 0.5]])
def test_market_weights_must_be_a_non_empty_series(
    covariance: pd.DataFrame,
    market_weights: object,
) -> None:
    with pytest.raises(ValueError, match="non-empty Series"):
        implied_equilibrium_returns(
            market_weights,  # type: ignore[arg-type]
            covariance,
            risk_aversion=2.5,
        )


def test_invalid_covariance_is_rejected() -> None:
    market_weights = pd.Series({"A": 0.50, "B": 0.50})
    covariance = pd.DataFrame(
        [[0.04, 0.02], [0.01, 0.09]],
        index=["A", "B"],
        columns=["A", "B"],
    )

    with pytest.raises(ValueError, match="symmetric"):
        implied_equilibrium_returns(
            market_weights,
            covariance,
            risk_aversion=2.5,
        )
