"""Benchmark comparison helpers for QuantAlloc."""

from collections.abc import Hashable, Mapping

import numpy as np
import pandas as pd

from src.metrics import (
    annualized_volatility,
    cagr,
    max_drawdown,
    sharpe_ratio,
    wealth_index,
)
from src.portfolio import calculate_asset_returns


def calculate_benchmark_returns(
    prices: pd.DataFrame,
    ticker: str,
) -> pd.Series:
    """Calculate daily returns for one benchmark ticker."""
    if ticker not in prices.columns:
        raise ValueError(f"benchmark ticker is missing from prices: {ticker}")

    benchmark_returns = calculate_asset_returns(prices[[ticker]])[ticker]
    benchmark_returns.name = ticker
    return benchmark_returns


def align_returns(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Align portfolio and benchmark returns to common dates."""
    if not isinstance(portfolio_returns, pd.Series):
        raise TypeError("portfolio_returns must be a pandas Series")
    if not isinstance(benchmark_returns, pd.Series):
        raise TypeError("benchmark_returns must be a pandas Series")

    aligned = _align_return_series(
        {"portfolio": portfolio_returns, "benchmark": benchmark_returns},
        context="benchmark",
        minimum_error="portfolio and benchmark must share at least two return dates",
    )

    return aligned["portfolio"], aligned["benchmark"]


def _align_return_series(
    return_series: Mapping[Hashable, pd.Series],
    *,
    context: str,
    minimum_error: str,
) -> pd.DataFrame:
    """Trim boundary dates without dropping observations inside the overlap."""
    if not return_series:
        raise ValueError("return_series must not be empty")
    validated = {}
    for name, returns in return_series.items():
        if not isinstance(returns, pd.Series) or returns.empty:
            raise ValueError(f"returns for {name} must be a non-empty Series")
        if not isinstance(returns.index, pd.DatetimeIndex):
            raise TypeError(f"returns for {name} must use a DatetimeIndex")
        if returns.index.has_duplicates:
            raise ValueError(f"returns index for {name} must be unique")
        if returns.index.hasnans:
            raise ValueError(f"returns index for {name} must not contain NaT")
        try:
            numeric = pd.to_numeric(returns, errors="raise").astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"returns for {name} must be numeric") from exc
        validated[name] = numeric.sort_index()

    start_date = max(returns.index[0] for returns in validated.values())
    end_date = min(returns.index[-1] for returns in validated.values())
    aligned = pd.concat(validated, axis=1, join="outer", sort=True)
    aligned = aligned.loc[
        (aligned.index >= start_date) & (aligned.index <= end_date)
    ]
    if len(aligned) < 2:
        raise ValueError(minimum_error)
    if aligned.isna().any().any():
        raise ValueError(
            f"aligned {context} returns must not contain NaN or missing dates "
            "within the common period"
        )
    if not np.isfinite(aligned.to_numpy()).all():
        raise ValueError(f"aligned {context} returns must contain only finite values")
    return aligned


def _metric_summary(
    returns: pd.Series,
    risk_free_rate: float,
    periods_per_year: int,
) -> dict[str, float]:
    return {
        "cagr": cagr(returns),
        "volatility": annualized_volatility(returns, periods_per_year),
        "sharpe": sharpe_ratio(
            returns,
            risk_free_rate,
            periods_per_year,
        ),
        "max_drawdown": max_drawdown(returns),
    }


def compare_with_benchmark(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> dict[str, dict[str, float] | pd.DataFrame]:
    """Compare portfolio and benchmark over their common date range."""
    portfolio, benchmark = align_returns(
        portfolio_returns,
        benchmark_returns,
    )

    cumulative_returns = pd.DataFrame(
        {
            "portfolio": wealth_index(portfolio) - 1.0,
            "benchmark": wealth_index(benchmark) - 1.0,
        }
    )

    return {
        "portfolio": _metric_summary(
            portfolio,
            risk_free_rate,
            periods_per_year,
        ),
        "benchmark": _metric_summary(
            benchmark,
            risk_free_rate,
            periods_per_year,
        ),
        "cumulative_returns": cumulative_returns,
    }
