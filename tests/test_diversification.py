"""Tests for diversification analysis."""

import pandas as pd
import pytest

from src.diversification import (
    correlation_matrix,
    covariance_matrix,
    effective_number_of_assets,
    hhi,
)


@pytest.fixture
def asset_returns() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "A": [0.01, 0.02, -0.01, 0.03],
            "B": [0.02, 0.01, 0.00, 0.04],
            "C": [-0.01, 0.01, 0.02, -0.02],
        }
    )


def test_correlation_matrix_matches_pandas(
    asset_returns: pd.DataFrame,
) -> None:
    result = correlation_matrix(asset_returns)

    pd.testing.assert_frame_equal(result, asset_returns.corr())


def test_covariance_matrix_is_annualized(
    asset_returns: pd.DataFrame,
) -> None:
    result = covariance_matrix(asset_returns)

    pd.testing.assert_frame_equal(result, asset_returns.cov() * 252)


def test_covariance_matrix_accepts_custom_periods(
    asset_returns: pd.DataFrame,
) -> None:
    result = covariance_matrix(asset_returns, periods_per_year=12)

    pd.testing.assert_frame_equal(result, asset_returns.cov() * 12)


def test_covariance_matrix_rejects_invalid_periods(
    asset_returns: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        covariance_matrix(asset_returns, periods_per_year=0)


def test_hhi_for_four_equal_weights() -> None:
    weights = pd.Series([0.25, 0.25, 0.25, 0.25])

    assert hhi(weights) == pytest.approx(0.25)


def test_effective_number_for_four_equal_weights() -> None:
    weights = pd.Series([0.25, 0.25, 0.25, 0.25])

    assert effective_number_of_assets(weights) == pytest.approx(4.0)


def test_effective_number_decreases_for_concentrated_weights() -> None:
    weights = pd.Series([0.60, 0.20, 0.10, 0.10])

    assert hhi(weights) == pytest.approx(0.42)
    assert effective_number_of_assets(weights) == pytest.approx(1 / 0.42)
