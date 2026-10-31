"""Single-market-factor exposure using aligned daily OOS returns."""

import numpy as np
import pandas as pd


def market_factor_exposure(
    returns: pd.DataFrame,
    benchmark: str = "TOPIX",
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Estimate OLS beta, annualized daily intercept and R-squared.

    Model: strategy_daily = alpha_daily + beta * benchmark_daily + error.

    Alpha is annualized arithmetically as alpha_daily * periods_per_year.
    It is not compounded alpha, causal attribution, or evidence of skill.
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a DataFrame")
    if returns.empty or len(returns) < 2:
        raise ValueError("returns must contain at least two observations")
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("returns must use a DatetimeIndex")
    if (
        returns.index.has_duplicates
        or returns.index.hasnans
        or not returns.index.is_monotonic_increasing
    ):
        raise ValueError("returns dates must be sorted and unique")
    if returns.columns.has_duplicates:
        raise ValueError("returns columns must be unique")
    if not isinstance(benchmark, str) or benchmark not in returns.columns:
        raise ValueError("benchmark column is missing")
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
        raise ValueError("returns must be finite")
    if (validated < -1.0).any().any():
        raise ValueError("returns must not be below -100%")

    market = validated[benchmark].to_numpy()
    market_mean = float(market.mean())
    market_centered = market - market_mean

    market_variation = float(np.dot(market_centered, market_centered))
    market_energy = float(np.dot(market, market))
    tolerance = np.finfo(float).eps

    rows = {}

    for name in validated.columns:
        strategy = validated[name].to_numpy()
        strategy_mean = float(strategy.mean())
        strategy_centered = strategy - strategy_mean

        # No meaningful slope when the market does not vary.
        if market_variation <= tolerance * market_energy:
            rows[name] = {
                "beta": np.nan,
                "alpha_annual": np.nan,
                "r_squared": np.nan,
            }
            continue

        beta = float(np.dot(market_centered, strategy_centered) / market_variation)
        alpha_daily = strategy_mean - beta * market_mean

        strategy_variation = float(np.dot(strategy_centered, strategy_centered))
        strategy_energy = float(np.dot(strategy, strategy))

        if strategy_variation <= tolerance * strategy_energy:
            r_squared = np.nan
        else:
            residual = strategy_centered - beta * market_centered
            unexplained = float(np.dot(residual, residual))
            r_squared = float(
                np.clip(
                    1.0 - unexplained / strategy_variation,
                    0.0,
                    1.0,
                )
            )

        rows[name] = {
            "beta": beta,
            "alpha_annual": alpha_daily * periods_per_year,
            "r_squared": r_squared,
        }

    result = pd.DataFrame.from_dict(rows, orient="index")
    result.index.name = "Strategy"

    return result
