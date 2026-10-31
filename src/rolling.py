"""Rolling risk metrics on aligned out-of-sample daily returns."""

import math

import numpy as np
import pandas as pd


def rolling_risk_metrics(
    returns: pd.DataFrame,
    window: int,
    periods_per_year: int = 252,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return annualized rolling volatility and Sharpe (risk-free rate 0%)."""
    if not isinstance(returns, pd.DataFrame) or returns.empty:
        raise ValueError("returns must be a non-empty DataFrame")

    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("returns must use a DatetimeIndex")

    if not returns.index.is_monotonic_increasing or returns.index.has_duplicates:
        raise ValueError("returns dates must be sorted and unique")

    if returns.columns.has_duplicates:
        raise ValueError("returns columns must be unique")

    if isinstance(window, bool) or not isinstance(window, int) or window < 2:
        raise ValueError("window must be an integer >= 2")

    if (
        isinstance(periods_per_year, bool)
        or not isinstance(periods_per_year, int)
        or periods_per_year <= 0
    ):
        raise ValueError("periods_per_year must be a positive integer")

    try:
        validated = returns.apply(pd.to_numeric, errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("returns must be numeric") from exc

    if not np.isfinite(validated.to_numpy()).all():
        raise ValueError("returns must contain finite values")

    if (validated < -1.0).any().any():
        raise ValueError("returns must not be below -100%")

    rolling = validated.rolling(window=window, min_periods=window)

    volatility = rolling.std(ddof=1) * math.sqrt(periods_per_year)

    annualized_mean = rolling.mean() * periods_per_year

    # Sharpe is undefined when volatility is effectively zero.
    sharpe = annualized_mean / volatility.where(volatility > 1e-12)

    return volatility, sharpe
