"""Correlation-clustered hierarchical risk parity (HRP).

Single-linkage clustering, quasi-diagonal ordering, and recursive bisection
with inverse-variance cluster portfolios. Uses covariance available at the
rebalance date, without future observations or expected-return forecasts.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from src.risk import validate_covariance


def _cluster_variance(matrix: np.ndarray, indices: np.ndarray) -> float:
    """Variance of an inverse-variance-weighted sub-portfolio."""
    submatrix = matrix[np.ix_(indices, indices)]
    diagonal = np.diag(submatrix)
    # Scaling by the smallest variance avoids taking reciprocals of tiny values.
    inverse_variance = diagonal.min() / diagonal
    inverse_variance /= inverse_variance.sum()
    variance = float(inverse_variance @ submatrix @ inverse_variance)
    if variance < -1e-10:
        raise ValueError("cluster variance must not be negative")
    return max(0.0, variance)


def hierarchical_risk_parity_weights(covariance: pd.DataFrame) -> pd.Series:
    """Return fully invested, long-only HRP weights in original asset order."""
    validated = validate_covariance(covariance)
    matrix = validated.to_numpy(dtype=float)
    count = len(validated)
    variances = np.diag(matrix)
    if (variances <= 0.0).any():
        raise ValueError("covariance must contain positive asset variances")
    if count == 1:
        return pd.Series(
            [1.0], index=validated.columns, name="hierarchical_risk_parity"
        )

    volatility = np.sqrt(variances)
    correlation = matrix / np.outer(volatility, volatility)
    correlation = (correlation + correlation.T) / 2.0
    np.clip(correlation, -1.0, 1.0, out=correlation)
    np.fill_diagonal(correlation, 1.0)

    distance = np.sqrt(np.maximum(0.0, (1.0 - correlation) / 2.0))
    np.fill_diagonal(distance, 0.0)
    tree = linkage(
        squareform(distance, checks=True),
        method="single",
        optimal_ordering=False,
    )
    order = leaves_list(tree)
    weights = np.ones(count, dtype=float)
    remaining = [order]

    while remaining:
        cluster = remaining.pop()
        if len(cluster) == 1:
            continue
        split = len(cluster) // 2
        left, right = cluster[:split], cluster[split:]
        left_variance = _cluster_variance(matrix, left)
        right_variance = _cluster_variance(matrix, right)
        total = left_variance + right_variance
        left_fraction = right_variance / total if total > 0.0 else 0.5
        weights[left] *= left_fraction
        weights[right] *= 1.0 - left_fraction
        remaining.extend((left, right))

    if not np.isfinite(weights).all() or weights.sum() <= 0.0:
        raise RuntimeError("hierarchical allocation returned invalid weights")
    weights /= weights.sum()
    return pd.Series(weights, index=validated.columns, name="hierarchical_risk_parity")
