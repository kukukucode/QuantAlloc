"""Diversification analysis for QuantAlloc portfolios."""

import numpy as np
import pandas as pd

from src.portfolio import validate_weights


def _validate_asset_returns(asset_returns: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(asset_returns, pd.DataFrame) or asset_returns.empty:
        raise ValueError("asset_returns must be a non-empty DataFrame")
    if len(asset_returns) < 2:
        raise ValueError("asset_returns must contain at least two observations")
    if asset_returns.columns.has_duplicates:
        raise ValueError("asset_returns columns must be unique")

    try:
        validated = asset_returns.apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("asset_returns must be numeric") from exc

    if validated.isna().any().any():
        raise ValueError("asset_returns must not contain missing values")
    if not np.isfinite(validated.to_numpy(dtype=float)).all():
        raise ValueError("asset_returns must contain only finite values")

    return validated


def correlation_matrix(asset_returns: pd.DataFrame) -> pd.DataFrame:
    """Calculate the correlation matrix of daily asset returns."""
    validated = _validate_asset_returns(asset_returns)
    return validated.corr()


def covariance_matrix(
    asset_returns: pd.DataFrame,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Calculate the annualized covariance matrix of asset returns."""
    if isinstance(periods_per_year, bool) or not isinstance(
        periods_per_year,
        int,
    ):
        raise ValueError("periods_per_year must be a positive integer")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be a positive integer")

    validated = _validate_asset_returns(asset_returns)
    return validated.cov() * periods_per_year


def hhi(weights: pd.Series) -> float:
    """Calculate the Herfindahl-Hirschman concentration index."""
    if not isinstance(weights, pd.Series):
        raise TypeError("weights must be a pandas Series")

    validated = validate_weights(weights.to_dict(), list(weights.index))
    return float(validated.pow(2).sum())


def effective_number_of_assets(weights: pd.Series) -> float:
    """Calculate the effective number of assets as the inverse HHI."""
    return 1.0 / hhi(weights)
