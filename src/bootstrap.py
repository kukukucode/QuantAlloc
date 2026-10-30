"""Bootstrap resampling utilities."""

import numpy as np
import pandas as pd
from src.portfolio import calculate_portfolio_returns
from src.evaluation import performance_summary


def bootstrap_indices(
    n_observations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample observation indices with replacement."""
    if (
        isinstance(n_observations, bool)
        or not isinstance(n_observations, int)
        or n_observations <= 0
    ):
        raise ValueError("n_observations must be a positive integer")

    return rng.integers(
        low=0,
        high=n_observations,
        size=n_observations,
    )


def resample_asset_returns(
    asset_returns: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Resample daily asset returns with replacement."""
    if not isinstance(asset_returns, pd.DataFrame) or asset_returns.empty:
        raise ValueError("asset_returns must be a non-empty DataFrame")

    if not isinstance(asset_returns.index, pd.DatetimeIndex):
        raise TypeError("asset_returns must use a DatetimeIndex")

    if not asset_returns.index.is_monotonic_increasing:
        raise ValueError("asset_returns index must be sorted")

    if asset_returns.index.has_duplicates:
        raise ValueError("asset_returns dates must be unique")

    if asset_returns.columns.has_duplicates:
        raise ValueError("asset_returns columns must be unique")

    try:
        validated = asset_returns.apply(
            pd.to_numeric,
            errors="raise",
        ).astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("asset_returns must be numeric") from exc

    if not np.isfinite(validated.to_numpy()).all():
        raise ValueError("asset_returns must contain finite values")

    indices = bootstrap_indices(len(validated), rng)

    resampled = validated.iloc[indices].copy()
    resampled.index = validated.index

    return resampled


def bootstrap_portfolio_metrics(
    asset_returns: pd.DataFrame,
    weights: pd.Series,
    rng: np.random.Generator,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """Calculate portfolio metrics for one bootstrap sample."""
    sampled_returns = resample_asset_returns(
        asset_returns,
        rng,
    )

    portfolio_returns = calculate_portfolio_returns(
        sampled_returns,
        weights,
    )

    return performance_summary(
        portfolio_returns,
        risk_free_rate=risk_free_rate,
    )


def bootstrap_metric_distribution(
    asset_returns: pd.DataFrame,
    weights: pd.Series,
    n_bootstrap: int = 1000,
    seed: int = 42,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """Estimate the distribution of portfolio metrics using bootstrap."""
    if (
        isinstance(n_bootstrap, bool)
        or not isinstance(n_bootstrap, int)
        or n_bootstrap <= 0
    ):
        raise ValueError("n_bootstrap must be a positive integer")

    rng = np.random.default_rng(seed)

    results = [
        bootstrap_portfolio_metrics(
            asset_returns,
            weights,
            rng,
            risk_free_rate=risk_free_rate,
        )
        for _ in range(n_bootstrap)
    ]

    distribution = pd.DataFrame(results)
    distribution.index.name = "bootstrap_sample"

    return distribution


def bootstrap_metric_intervals(
    distribution: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate percentile ranges for bootstrap metrics."""
    if not isinstance(distribution, pd.DataFrame) or distribution.empty:
        raise ValueError("distribution must be a non-empty DataFrame")

    intervals = distribution.quantile([0.025, 0.5, 0.975]).T

    intervals.columns = [
        "lower_2_5",
        "median",
        "upper_97_5",
    ]

    return intervals


def bootstrap_metric_intervals(
    distribution: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate percentile ranges for bootstrap metrics."""
    if not isinstance(distribution, pd.DataFrame) or distribution.empty:
        raise ValueError("distribution must be a non-empty DataFrame")

    intervals = distribution.quantile([0.025, 0.5, 0.975]).T

    intervals.columns = [
        "lower_2_5",
        "median",
        "upper_97_5",
    ]

    return intervals


def bootstrap_portfolio_summary(
    asset_returns: pd.DataFrame,
    weights: pd.Series,
    n_bootstrap: int = 1000,
    seed: int = 42,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """Compare observed metrics with bootstrap percentile ranges."""
    distribution = bootstrap_metric_distribution(
        asset_returns,
        weights,
        n_bootstrap=n_bootstrap,
        seed=seed,
        risk_free_rate=risk_free_rate,
    )

    intervals = bootstrap_metric_intervals(distribution)

    observed = performance_summary(
        calculate_portfolio_returns(asset_returns, weights),
        risk_free_rate=risk_free_rate,
    )

    intervals.insert(0, "observed", pd.Series(observed))

    return intervals
