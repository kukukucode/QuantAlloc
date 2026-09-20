"""Core Black-Litterman calculations."""

import math

import numpy as np
import pandas as pd

from src.portfolio import validate_weights
from src.risk import validate_covariance


def validate_views(
    pick_matrix: pd.DataFrame,
    view_returns: pd.Series,
    view_uncertainty: pd.DataFrame,
    assets: pd.Index | list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Validate and align the Black-Litterman P, Q, and Omega inputs."""
    asset_index = pd.Index(assets)
    if asset_index.empty:
        raise ValueError("assets must not be empty")
    if asset_index.has_duplicates:
        raise ValueError("assets must be unique")

    if not isinstance(pick_matrix, pd.DataFrame) or pick_matrix.empty:
        raise ValueError("pick_matrix must be a non-empty DataFrame")
    if pick_matrix.index.has_duplicates:
        raise ValueError("pick_matrix index must be unique")
    if pick_matrix.columns.has_duplicates:
        raise ValueError("pick_matrix columns must be unique")
    if set(pick_matrix.columns) != set(asset_index):
        raise ValueError("pick_matrix columns and assets must match")

    if not isinstance(view_returns, pd.Series) or view_returns.empty:
        raise ValueError("view_returns must be a non-empty Series")
    if view_returns.index.has_duplicates:
        raise ValueError("view_returns index must be unique")
    if set(view_returns.index) != set(pick_matrix.index):
        raise ValueError("pick_matrix and view_returns views must match")

    if (
        not isinstance(view_uncertainty, pd.DataFrame)
        or view_uncertainty.empty
    ):
        raise ValueError("view_uncertainty must be a non-empty DataFrame")
    if view_uncertainty.shape[0] != view_uncertainty.shape[1]:
        raise ValueError("view_uncertainty must be square")
    if view_uncertainty.index.has_duplicates:
        raise ValueError("view_uncertainty index must be unique")
    if view_uncertainty.columns.has_duplicates:
        raise ValueError("view_uncertainty columns must be unique")
    if set(view_uncertainty.index) != set(pick_matrix.index) or set(
        view_uncertainty.columns
    ) != set(pick_matrix.index):
        raise ValueError(
            "pick_matrix and view_uncertainty views must match"
        )

    try:
        validated_pick_matrix = pick_matrix.apply(
            pd.to_numeric,
            errors="raise",
        ).astype(float)
        validated_view_returns = pd.to_numeric(
            view_returns,
            errors="raise",
        ).astype(float)
        validated_view_uncertainty = view_uncertainty.apply(
            pd.to_numeric,
            errors="raise",
        ).astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("P, Q, and Omega must be numeric") from exc

    validated_pick_matrix = validated_pick_matrix.reindex(
        columns=asset_index
    )
    view_index = validated_pick_matrix.index
    validated_view_returns = validated_view_returns.reindex(view_index)
    validated_view_uncertainty = validated_view_uncertainty.reindex(
        index=view_index,
        columns=view_index,
    )

    if not np.isfinite(validated_pick_matrix.to_numpy()).all():
        raise ValueError("pick_matrix must contain only finite values")
    if not np.isfinite(validated_view_returns.to_numpy()).all():
        raise ValueError("view_returns must contain only finite values")
    if not np.isfinite(validated_view_uncertainty.to_numpy()).all():
        raise ValueError("view_uncertainty must contain only finite values")
    if np.any(np.all(validated_pick_matrix.to_numpy() == 0.0, axis=1)):
        raise ValueError("each view must contain at least one non-zero exposure")

    uncertainty_values = validated_view_uncertainty.to_numpy()
    if not np.allclose(uncertainty_values, uncertainty_values.T):
        raise ValueError("view_uncertainty must be symmetric")
    if np.any(np.linalg.eigvalsh(uncertainty_values) <= 0.0):
        raise ValueError("view_uncertainty must be positive definite")

    return (
        validated_pick_matrix,
        validated_view_returns,
        validated_view_uncertainty,
    )


def implied_equilibrium_returns(
    market_weights: pd.Series,
    covariance: pd.DataFrame,
    risk_aversion: float,
) -> pd.Series:
    """Calculate market equilibrium returns as risk_aversion * covariance * weights."""
    if not isinstance(market_weights, pd.Series) or market_weights.empty:
        raise ValueError("market_weights must be a non-empty Series")
    if market_weights.index.has_duplicates:
        raise ValueError("market_weights index must be unique")
    if isinstance(risk_aversion, bool):
        raise ValueError("risk_aversion must be a positive finite number")
    try:
        validated_risk_aversion = float(risk_aversion)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "risk_aversion must be a positive finite number"
        ) from exc
    if (
        not math.isfinite(validated_risk_aversion)
        or validated_risk_aversion <= 0.0
    ):
        raise ValueError("risk_aversion must be a positive finite number")

    validated_covariance = validate_covariance(covariance)
    if set(market_weights.index) != set(validated_covariance.columns):
        raise ValueError("market_weights and covariance assets must match")

    validated_weights = validate_weights(
        market_weights.to_dict(),
        list(validated_covariance.columns),
    ).reindex(validated_covariance.columns)
    implied_returns = (
        validated_covariance.dot(validated_weights)
        * validated_risk_aversion
    )
    implied_returns.name = "implied_equilibrium_return"
    return implied_returns
