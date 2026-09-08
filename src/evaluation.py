"""Fair out-of-sample strategy evaluation and comparison."""

import numpy as np
import pandas as pd

from src.backtest import BacktestResult, walk_forward_backtest
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


def turnover_summary(turnover: pd.Series) -> dict[str, float]:
    """Summarize turnover across all recorded rebalance dates."""
    if not isinstance(turnover, pd.Series) or turnover.empty:
        raise ValueError("turnover must be a non-empty Series")
    try:
        validated = pd.to_numeric(turnover, errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("turnover must be numeric") from exc
    if not np.isfinite(validated.to_numpy()).all():
        raise ValueError("turnover must contain only finite values")
    if (validated < 0.0).any():
        raise ValueError("turnover must be non-negative")

    return {
        "average_turnover": float(validated.mean()),
        "total_turnover": float(validated.sum()),
        "maximum_turnover": float(validated.max()),
    }


def compare_turnover(
    strategy_turnover: dict[str, pd.Series],
) -> pd.DataFrame:
    """Compare turnover and weight stability across strategies."""
    if not strategy_turnover:
        raise ValueError("strategy_turnover must not be empty")

    comparison = pd.DataFrame(
        {
            name: turnover_summary(turnover)
            for name, turnover in strategy_turnover.items()
        }
    ).T
    comparison.index.name = "Strategy"
    return comparison[
        ["average_turnover", "total_turnover", "maximum_turnover"]
    ]


def compare_rebalancing_frequencies(
    asset_returns: pd.DataFrame,
    strategies: tuple[str, ...] = (
        "equal_weight",
        "minimum_variance",
        "maximum_sharpe",
    ),
    holding_periods: tuple[int, ...] = (21, 63, 126, 252),
    estimation_window: int = 504,
    risk_free_rate: float = 0.0,
    transaction_cost_rate: float = 0.001,
) -> pd.DataFrame:
    """Compare net OOS results across rebalancing frequencies."""
    if not strategies:
        raise ValueError("strategies must not be empty")
    if not holding_periods:
        raise ValueError("holding_periods must not be empty")
    if len(set(strategies)) != len(strategies):
        raise ValueError("strategies must not contain duplicates")
    if len(set(holding_periods)) != len(holding_periods):
        raise ValueError("holding_periods must not contain duplicates")

    results: dict[tuple[str, int], BacktestResult] = {}
    for strategy in strategies:
        for holding_period in holding_periods:
            results[(strategy, holding_period)] = walk_forward_backtest(
                asset_returns,
                strategy,
                estimation_window,
                holding_period,
                risk_free_rate,
                transaction_cost_rate,
            )

    aligned_net_returns = pd.concat(
        {
            key: result.net_returns
            for key, result in results.items()
        },
        axis=1,
        join="inner",
    ).sort_index()
    if len(aligned_net_returns) < 2:
        raise ValueError("frequency results must share at least two OOS dates")
    if aligned_net_returns.isna().any().any():
        raise ValueError("aligned frequency returns must not contain NaN")

    start_date = aligned_net_returns.index[0]
    end_date = aligned_net_returns.index[-1]
    rows: list[dict[str, float | int | str]] = []
    for key, result in results.items():
        strategy, holding_period = key
        summary = performance_summary(aligned_net_returns[key])
        turnover = result.turnover.loc[
            (result.turnover.index >= start_date)
            & (result.turnover.index <= end_date)
        ]
        turnover_metrics = turnover_summary(turnover)
        costs = result.transaction_costs.reindex(turnover.index)
        rows.append(
            {
                "strategy": strategy,
                "holding_period": holding_period,
                "rebalances": len(turnover),
                "net_cagr": summary["cagr"],
                "net_volatility": summary["volatility"],
                "net_sharpe": summary["sharpe"],
                "net_max_drawdown": summary["max_drawdown"],
                **turnover_metrics,
                "total_transaction_cost": float(costs.sum()),
            }
        )

    comparison = pd.DataFrame(rows).set_index(
        ["strategy", "holding_period"]
    )
    comparison.attrs = {
        "start_date": start_date,
        "end_date": end_date,
        "observations": len(aligned_net_returns),
        "transaction_cost_rate": transaction_cost_rate,
    }
    return comparison
