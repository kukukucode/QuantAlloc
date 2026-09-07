"""Portfolio risk and risk-contribution calculations."""

import math

import numpy as np
import pandas as pd

from src.portfolio import validate_weights


def _validate_covariance(covariance: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(covariance, pd.DataFrame) or covariance.empty:
        raise ValueError("covariance must be a non-empty DataFrame")
    if covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be square")
    if not covariance.index.equals(covariance.columns):
        raise ValueError("covariance index and columns must match")

    try:
        validated = covariance.apply(pd.to_numeric, errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("covariance must be numeric") from exc

    values = validated.to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("covariance must contain only finite values")
    if not np.allclose(values, values.T, atol=1e-10):
        raise ValueError("covariance must be symmetric")
    if np.linalg.eigvalsh(values).min() < -1e-10:
        raise ValueError("covariance must be positive semidefinite")

    return validated


def _align_weights(
    weights: pd.Series,
    covariance: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame]:
    if not isinstance(weights, pd.Series):
        raise TypeError("weights must be a pandas Series")

    validated_covariance = _validate_covariance(covariance)
    if set(weights.index) != set(validated_covariance.columns):
        raise ValueError("weights and covariance assets must match")

    validated_weights = validate_weights(
        weights.to_dict(),
        list(validated_covariance.columns),
    ).reindex(validated_covariance.columns)
    return validated_weights, validated_covariance


def portfolio_volatility(
    weights: pd.Series,
    covariance: pd.DataFrame,
) -> float:
    """Calculate annualized portfolio volatility."""
    aligned_weights, validated_covariance = _align_weights(
        weights,
        covariance,
    )
    values = aligned_weights.to_numpy()
    variance = float(values @ validated_covariance.to_numpy() @ values)
    if variance < -1e-12:
        raise ValueError("portfolio variance must not be negative")

    return math.sqrt(max(variance, 0.0))


def risk_contribution(
    weights: pd.Series,
    covariance: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate marginal, component, and percentage risk contributions."""
    aligned_weights, validated_covariance = _align_weights(
        weights,
        covariance,
    )
    volatility = portfolio_volatility(
        aligned_weights,
        validated_covariance,
    )
    if np.isclose(volatility, 0.0):
        raise ValueError("risk contribution is undefined at zero volatility")

    covariance_times_weights = (
        validated_covariance.to_numpy() @ aligned_weights.to_numpy()
    )
    marginal = covariance_times_weights / volatility
    component = aligned_weights.to_numpy() * marginal
    percentage = component / volatility

    return pd.DataFrame(
        {
            "weight": aligned_weights,
            "marginal_risk": marginal,
            "risk_contribution": component,
            "risk_contribution_pct": percentage,
        },
        index=aligned_weights.index,
    )
