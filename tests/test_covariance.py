"""Tests for covariance estimators."""

import numpy as np
import pandas as pd
import pytest

from src.covariance import estimate_covariance
from src.diversification import covariance_matrix


@pytest.fixture
def asset_returns() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "A": [0.01, 0.02, -0.01, 0.03, 0.00],
            "B": [0.02, 0.01, 0.00, 0.04, -0.01],
            "C": [-0.01, 0.01, 0.02, -0.02, 0.03],
        }
    )


def test_sample_matches_existing_covariance_matrix(
    asset_returns: pd.DataFrame,
) -> None:
    result = estimate_covariance(asset_returns, method="sample")

    pd.testing.assert_frame_equal(result, covariance_matrix(asset_returns))


def test_ledoit_wolf_preserves_labels_and_is_valid(
    asset_returns: pd.DataFrame,
) -> None:
    result = estimate_covariance(asset_returns, method="ledoit_wolf")

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (3, 3)
    assert result.index.to_list() == ["A", "B", "C"]
    assert result.columns.to_list() == ["A", "B", "C"]
    assert np.isfinite(result.to_numpy()).all()
    assert np.allclose(result.to_numpy(), result.to_numpy().T)
    assert np.linalg.eigvalsh(result.to_numpy()).min() >= -1e-10


@pytest.mark.parametrize("method", ["sample", "ledoit_wolf"])
def test_annualization_uses_periods_per_year(
    asset_returns: pd.DataFrame,
    method: str,
) -> None:
    monthly = estimate_covariance(
        asset_returns,
        method=method,
        periods_per_year=12,
    )
    annual = estimate_covariance(
        asset_returns,
        method=method,
        periods_per_year=252,
    )

    pd.testing.assert_frame_equal(annual, monthly * (252 / 12))


def test_unknown_method_is_rejected(asset_returns: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="unknown covariance method: magic"):
        estimate_covariance(asset_returns, method="magic")


@pytest.mark.parametrize("periods_per_year", [0, -1, 12.5, True])
def test_invalid_periods_per_year_is_rejected(
    asset_returns: pd.DataFrame,
    periods_per_year: object,
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        estimate_covariance(
            asset_returns,
            periods_per_year=periods_per_year,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("invalid_value", "message"),
    [(np.nan, "missing values"), (np.inf, "finite values")],
)
def test_invalid_values_are_rejected(
    asset_returns: pd.DataFrame,
    invalid_value: float,
    message: str,
) -> None:
    asset_returns.iloc[0, 0] = invalid_value

    with pytest.raises(ValueError, match=message):
        estimate_covariance(asset_returns)


def test_duplicate_assets_are_rejected(asset_returns: pd.DataFrame) -> None:
    asset_returns.columns = ["A", "A", "C"]

    with pytest.raises(ValueError, match="columns must be unique"):
        estimate_covariance(asset_returns)


def test_at_least_two_observations_are_required() -> None:
    with pytest.raises(ValueError, match="at least two observations"):
        estimate_covariance(pd.DataFrame({"A": [0.01], "B": [0.02]}))
