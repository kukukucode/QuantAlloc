"""Long-only equal risk contribution portfolio optimization."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.risk import risk_contribution, validate_covariance


def equal_risk_contribution_weights(
    covariance: pd.DataFrame,
) -> pd.Series:
    """Find fully invested long-only weights with equal risk contributions."""
    validated_covariance = validate_covariance(covariance)
    matrix = validated_covariance.to_numpy()
    asset_count = len(validated_covariance)
    target_contribution = np.full(asset_count, 1.0 / asset_count)

    asset_variances = np.diag(matrix)
    if (asset_variances <= 0.0).any():
        raise ValueError("covariance must contain positive asset variances")
    diagonal_scale = float(np.mean(asset_variances))
    scaled_matrix = matrix / diagonal_scale

    def risk_budget_objective(weights: np.ndarray) -> float:
        return float(
            0.5 * weights @ scaled_matrix @ weights
            - target_contribution @ np.log(weights)
        )

    def risk_budget_gradient(weights: np.ndarray) -> np.ndarray:
        return scaled_matrix @ weights - target_contribution / weights

    primary_result = minimize(
        risk_budget_objective,
        np.ones(asset_count),
        method="L-BFGS-B",
        jac=risk_budget_gradient,
        bounds=[(1e-12, None)] * asset_count,
        options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 2000},
    )

    def validated_result(values: np.ndarray) -> pd.Series | None:
        if not np.isfinite(values).all():
            return None
        weights = np.maximum(values, 0.0)
        total_weight = float(weights.sum())
        if total_weight <= 0.0:
            return None
        candidate = pd.Series(
            weights / total_weight,
            index=validated_covariance.columns,
            name="risk_parity",
        )
        contributions = risk_contribution(
            candidate,
            validated_covariance,
        )["risk_contribution_pct"]
        if np.allclose(
            contributions.to_numpy(),
            target_contribution,
            atol=1e-5,
            rtol=0.0,
        ):
            return candidate
        return None

    primary_weights = validated_result(primary_result.x)
    if primary_weights is not None:
        return primary_weights

    fallback_initial = (
        primary_result.x
        if np.isfinite(primary_result.x).all()
        and (primary_result.x > 0.0).all()
        else np.ones(asset_count)
    )
    fallback_result = minimize(
        risk_budget_objective,
        fallback_initial,
        method="SLSQP",
        jac=risk_budget_gradient,
        bounds=[(1e-12, None)] * asset_count,
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    fallback_weights = validated_result(fallback_result.x)
    if fallback_weights is not None:
        return fallback_weights

    raise RuntimeError(
        "risk parity optimization did not achieve equal risk contributions; "
        f"primary: {primary_result.message}; "
        f"fallback: {fallback_result.message}"
    )
