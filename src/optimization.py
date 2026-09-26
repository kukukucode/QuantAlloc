"""Classical long-only mean-variance portfolio optimization."""

import math
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.portfolio import validate_weights
from src.risk import portfolio_volatility, validate_covariance


def expected_returns(
    asset_returns: pd.DataFrame,
    periods_per_year: int = 252,
) -> pd.Series:
    """Annualize historical arithmetic mean returns."""
    if not isinstance(asset_returns, pd.DataFrame) or asset_returns.empty:
        raise ValueError("asset_returns must be a non-empty DataFrame")
    if isinstance(periods_per_year, bool) or not isinstance(
        periods_per_year,
        int,
    ):
        raise ValueError("periods_per_year must be a positive integer")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be a positive integer")

    try:
        validated = asset_returns.apply(pd.to_numeric, errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("asset_returns must be numeric") from exc
    if validated.isna().any().any():
        raise ValueError("asset_returns must not contain missing values")
    if not np.isfinite(validated.to_numpy()).all():
        raise ValueError("asset_returns must contain only finite values")

    result = validated.mean() * periods_per_year
    result.name = "expected_return"
    return result


def _validate_optimization_inputs(
    returns: pd.Series,
    covariance: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame]:
    if not isinstance(returns, pd.Series) or returns.empty:
        raise ValueError("expected_returns must be a non-empty Series")
    if returns.index.has_duplicates:
        raise ValueError("expected_returns index must be unique")

    validated_covariance = validate_covariance(covariance)
    if set(returns.index) != set(validated_covariance.columns):
        raise ValueError("expected_returns and covariance assets must match")

    try:
        validated_returns = pd.to_numeric(returns, errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("expected_returns must be numeric") from exc
    validated_returns = validated_returns.reindex(validated_covariance.columns)
    if not np.isfinite(validated_returns.to_numpy()).all():
        raise ValueError("expected_returns must contain only finite values")

    return validated_returns, validated_covariance


def _solve(
    objective: Callable[[np.ndarray], float],
    asset_count: int,
    constraints: list[dict[str, object]],
    initial_weights: np.ndarray | None = None,
    gradient: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    initial = (
        np.full(asset_count, 1.0 / asset_count)
        if initial_weights is None
        else initial_weights
    )
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        jac=gradient,
        bounds=[(0.0, 1.0)] * asset_count,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"portfolio optimization failed: {result.message}")
    if not np.isfinite(result.x).all():
        raise RuntimeError("portfolio optimization returned non-finite weights")
    if (result.x < -1e-8).any() or (result.x > 1.0 + 1e-8).any():
        raise RuntimeError("portfolio optimization violated weight bounds")

    weights = np.clip(result.x, 0.0, 1.0)
    total_weight = float(weights.sum())
    if total_weight <= 0.0 or not np.isclose(total_weight, 1.0, atol=1e-8, rtol=0.0):
        raise RuntimeError(
            "portfolio optimization violated the fully-invested constraint"
        )
    weights = weights / total_weight
    for constraint in constraints:
        residual = float(constraint["fun"](weights))
        if not math.isfinite(residual) or abs(residual) > 1e-7:
            raise RuntimeError("portfolio optimization violated an equality constraint")
    if not math.isfinite(objective(weights)):
        raise RuntimeError("portfolio optimization returned a non-finite objective")
    return weights


def _scaled_covariance(matrix: np.ndarray) -> np.ndarray:
    """Remove the covariance unit scale from variance minimization."""
    scale = float(np.max(np.abs(matrix)))
    return matrix / scale if scale > 0.0 else matrix.copy()


def minimum_variance_weights(covariance: pd.DataFrame) -> pd.Series:
    """Find the fully invested long-only minimum-variance portfolio."""
    validated_covariance = validate_covariance(covariance)
    matrix = _scaled_covariance(validated_covariance.to_numpy())
    asset_count = len(validated_covariance)

    weights = _solve(
        lambda values: float(values @ matrix @ values),
        asset_count,
        [{"type": "eq", "fun": lambda values: values.sum() - 1.0}],
        gradient=lambda values: 2.0 * matrix @ values,
    )
    return pd.Series(
        weights,
        index=validated_covariance.columns,
        name="minimum_variance",
    )


def maximum_sharpe_weights(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> pd.Series:
    """Find the fully invested long-only maximum-Sharpe portfolio."""
    returns, validated_covariance = _validate_optimization_inputs(
        expected_returns,
        covariance,
    )
    if not math.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be finite")

    return_values = returns.to_numpy()
    matrix = validated_covariance.to_numpy()

    excess_returns = return_values - risk_free_rate
    if np.all(excess_returns <= 0.0):
        # For non-positive excess returns, diversification cannot improve
        # on the best individual Sharpe (the volatility triangle inequality).
        variances = np.diag(matrix)
        positive_variance = variances > 0.0
        if not positive_variance.any():
            raise ValueError("Sharpe ratio is undefined at zero volatility")
        asset_sharpes = np.full(len(returns), -np.inf)
        asset_sharpes[positive_variance] = (
            excess_returns[positive_variance]
            / np.sqrt(variances[positive_variance])
        )
        weights = np.zeros(len(returns))
        weights[int(np.argmax(asset_sharpes))] = 1.0
        return pd.Series(weights, index=returns.index, name="maximum_sharpe")

    def negative_sharpe(values: np.ndarray) -> float:
        variance = float(values @ matrix @ values)
        if variance <= 0.0:
            return float("inf")
        excess_return = float(values @ return_values) - risk_free_rate
        return -excess_return / math.sqrt(variance)

    weights = _solve(
        negative_sharpe,
        len(returns),
        [{"type": "eq", "fun": lambda values: values.sum() - 1.0}],
    )
    return pd.Series(
        weights,
        index=returns.index,
        name="maximum_sharpe",
    )


def portfolio_performance(
    weights: pd.Series,
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """Calculate expected return, volatility, and Sharpe ratio."""
    returns, validated_covariance = _validate_optimization_inputs(
        expected_returns,
        covariance,
    )
    if set(weights.index) != set(returns.index):
        raise ValueError("weights and expected_returns assets must match")

    aligned_weights = validate_weights(weights, list(returns.index)).reindex(
        returns.index
    )
    expected_return = float(aligned_weights @ returns)
    volatility = portfolio_volatility(
        aligned_weights,
        validated_covariance,
    )
    sharpe = (
        float("nan")
        if np.isclose(volatility, 0.0)
        else (expected_return - risk_free_rate) / volatility
    )
    return {
        "return": expected_return,
        "volatility": volatility,
        "sharpe": sharpe,
    }


def efficient_frontier(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    points: int = 30,
) -> pd.DataFrame:
    """Calculate long-only minimum-volatility portfolios by target return."""
    returns, validated_covariance = _validate_optimization_inputs(
        expected_returns,
        covariance,
    )
    if isinstance(points, bool) or not isinstance(points, int) or points < 2:
        raise ValueError("points must be an integer of at least 2")

    matrix = validated_covariance.to_numpy()
    return_values = returns.to_numpy()
    minimum_weights = minimum_variance_weights(validated_covariance)
    minimum_return = float(minimum_weights @ returns)
    maximum_return = float(returns.max())
    targets = np.linspace(minimum_return, maximum_return, points)

    if np.ptp(return_values) == 0.0:
        # The return constraint duplicates the fully-invested constraint.
        # Preserve the requested row count using the same efficient point.
        volatility = portfolio_volatility(minimum_weights, validated_covariance)
        return pd.DataFrame(
            {
                "return": [minimum_return] * points,
                "volatility": [volatility] * points,
            }
        )

    scaled_matrix = _scaled_covariance(matrix)
    return_floor = float(return_values.min())
    return_span = float(np.ptp(return_values))
    normalized_returns = (return_values - return_floor) / return_span

    rows: list[dict[str, float]] = []
    initial = minimum_weights.to_numpy()
    for target in targets:
        constraints = [
            {"type": "eq", "fun": lambda values: values.sum() - 1.0},
            {
                "type": "eq",
                "fun": lambda values, target=target: (
                    float(values @ normalized_returns)
                    - (target - return_floor) / return_span
                ),
            },
        ]
        weights = _solve(
            lambda values: float(values @ scaled_matrix @ values),
            len(returns),
            constraints,
            initial,
            gradient=lambda values: 2.0 * scaled_matrix @ values,
        )
        initial = weights
        rows.append(
            {
                "return": float(weights @ return_values),
                "volatility": math.sqrt(max(float(weights @ matrix @ weights), 0.0)),
            }
        )

    return pd.DataFrame(rows)
