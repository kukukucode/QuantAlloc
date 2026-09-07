"""Benchmark comparison helpers for QuantAlloc."""

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

    aligned = pd.concat(
        [
            portfolio_returns.rename("portfolio"),
            benchmark_returns.rename("benchmark"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    aligned = aligned.sort_index()

    if len(aligned) < 2:
        raise ValueError(
            "portfolio and benchmark must share at least two return dates"
        )

    return aligned["portfolio"], aligned["benchmark"]


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
