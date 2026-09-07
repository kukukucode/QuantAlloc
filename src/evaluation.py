"""Fair out-of-sample strategy evaluation and comparison."""

import numpy as np
import pandas as pd

from src.metrics import (
    annualized_volatility,
    cagr,
    max_drawdown,
    sharpe_ratio,
)


def performance_summary(returns: pd.Series) -> dict[str, float]:
    """Calculate the standard performance metrics for one return series."""
    return {
        "cagr": cagr(returns),
        "volatility": annualized_volatility(returns),
        "sharpe": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(returns),
    }


def _align_strategy_returns(
    strategy_returns: dict[str, pd.Series],
) -> pd.DataFrame:
    if not strategy_returns:
        raise ValueError("strategy_returns must not be empty")

    validated: dict[str, pd.Series] = {}
    for name, returns in strategy_returns.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("strategy names must be non-empty strings")
        if not isinstance(returns, pd.Series) or returns.empty:
            raise ValueError(f"returns for {name} must be a non-empty Series")
        if not isinstance(returns.index, pd.DatetimeIndex):
            raise TypeError(f"returns for {name} must use a DatetimeIndex")
        if returns.index.has_duplicates:
            raise ValueError(f"returns index for {name} must be unique")

        try:
            numeric = pd.to_numeric(returns, errors="raise").astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"returns for {name} must be numeric") from exc
        validated[name] = numeric.sort_index()

    aligned = pd.concat(validated, axis=1, join="inner").sort_index()
    if len(aligned) < 2:
        raise ValueError(
            "strategies must share at least two return dates"
        )
    if aligned.isna().any().any():
        raise ValueError("aligned strategy returns must not contain NaN")
    if not np.isfinite(aligned.to_numpy()).all():
        raise ValueError(
            "aligned strategy returns must contain only finite values"
        )

    return aligned


def compare_strategies(
    strategy_returns: dict[str, pd.Series],
) -> pd.DataFrame:
    """Compare strategies using metrics from their common date range."""
    aligned = _align_strategy_returns(strategy_returns)
    comparison = pd.DataFrame(
        {
            name: performance_summary(aligned[name])
            for name in aligned.columns
        }
    ).T
    comparison.index.name = "Strategy"
    comparison = comparison[
        ["cagr", "volatility", "sharpe", "max_drawdown"]
    ]
    comparison.attrs = {
        "start_date": aligned.index[0],
        "end_date": aligned.index[-1],
        "observations": len(aligned),
    }
    return comparison
