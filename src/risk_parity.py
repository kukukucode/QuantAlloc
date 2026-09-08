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

    result = minimize(
        risk_budget_objective,
        np.ones(asset_count),
        method="L-BFGS-B",
        jac=risk_budget_gradient,
        bounds=[(1e-12, None)] * asset_count,
        options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 2000},
    )
    if not result.success:
        raise RuntimeError(f"risk parity optimization failed: {result.message}")

    weights = np.maximum(result.x, 0.0)
    total_weight = float(weights.sum())
    if total_weight <= 0.0:
        raise RuntimeError("risk parity optimization returned zero total weight")

    erc_weights = pd.Series(
        weights / total_weight,
        index=validated_covariance.columns,
        name="risk_parity",
    )
    contributions = risk_contribution(
        erc_weights,
        validated_covariance,
    )["risk_contribution_pct"]
    if not np.allclose(
        contributions.to_numpy(),
        target_contribution,
        atol=1e-5,
        rtol=0.0,
    ):
        raise RuntimeError(
            "risk parity optimization did not achieve equal risk contributions"
        )

    return erc_weights
