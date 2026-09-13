"""Covariance estimators for portfolio construction."""

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


VALID_COVARIANCE_METHODS = frozenset({"sample", "ledoit_wolf"})


def _validate_inputs(
    asset_returns: pd.DataFrame,
    method: str,
    periods_per_year: int,
) -> pd.DataFrame:
    if method not in VALID_COVARIANCE_METHODS:
        raise ValueError(f"unknown covariance method: {method}")
    if isinstance(periods_per_year, bool) or not isinstance(
        periods_per_year,
        int,
    ):
        raise ValueError("periods_per_year must be a positive integer")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be a positive integer")
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


def estimate_covariance(
    asset_returns: pd.DataFrame,
    method: str = "sample",
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Estimate and annualize an asset covariance matrix."""
    validated = _validate_inputs(
        asset_returns,
        method,
        periods_per_year,
    )
    if method == "sample":
        return validated.cov() * periods_per_year

    model = LedoitWolf().fit(validated.to_numpy(dtype=float))
    return pd.DataFrame(
        model.covariance_ * periods_per_year,
        index=validated.columns,
        columns=validated.columns,
    )
