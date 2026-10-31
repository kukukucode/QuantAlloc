"""One-factor-at-a-time walk-forward parameter sensitivity."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtest import BacktestResult, walk_forward_backtest
from src.evaluation import _align_strategy_returns, performance_summary
from src.sample_config import (
    ESTIMATION_WINDOW,
    HOLDING_PERIOD,
    STRATEGY_LABELS,
    TRANSACTION_COST_RATE,
)


@dataclass(frozen=True)
class SensitivityResult:
    """Per-parameter metrics evaluated on common OOS dates."""

    strategy: str
    tables: dict[str, pd.DataFrame]


def _validate_options(parameter: str, options: tuple) -> None:
    if not isinstance(options, tuple) or not options:
        raise ValueError(f"{parameter} options must be a non-empty tuple")
    if len(options) != len(set(options)):
        raise ValueError(f"{parameter} options must be unique")

    for value in options:
        if parameter == "transaction_cost_rate":
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or not 0.0 <= value < 1.0
            ):
                raise ValueError("invalid transaction_cost_rate")
        elif (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < (2 if parameter == "estimation_window" else 1)
        ):
            raise ValueError(f"invalid {parameter}")


def run_parameter_sensitivity(
    asset_returns: pd.DataFrame,
    strategy: str,
    *,
    estimation_windows: tuple[int, ...] = (252, 378, 504),
    holding_periods: tuple[int, ...] = (21, 63, 126, 252),
    transaction_cost_rates: tuple[float, ...] = (0.0, 0.001, 0.003),
) -> SensitivityResult:
    """Change one parameter at a time, aligning OOS dates within each sweep.

    Infeasible scenarios are shown explicitly instead of silently replacing
    them with different parameter values. Identical configurations are cached.
    """
    if strategy not in STRATEGY_LABELS:
        raise ValueError(f"unsupported strategy: {strategy}")
    if not isinstance(asset_returns, pd.DataFrame):
        raise TypeError("asset_returns must be a DataFrame")
    if asset_returns.empty:
        raise ValueError("asset_returns must not be empty")

    sweeps = {
        "estimation_window": estimation_windows,
        "holding_period": holding_periods,
        "transaction_cost_rate": transaction_cost_rates,
    }

    for parameter, options in sweeps.items():
        _validate_options(parameter, options)

    baseline = {
        "estimation_window": ESTIMATION_WINDOW,
        "holding_period": HOLDING_PERIOD,
        "transaction_cost_rate": TRANSACTION_COST_RATE,
    }

    cache: dict[tuple[int, int, float], BacktestResult] = {}
    tables = {}

    for parameter, options in sweeps.items():
        scenarios = {}
        infeasible = []

        for value in options:
            settings = baseline.copy()
            settings[parameter] = value

            training = settings["estimation_window"]
            holding = settings["holding_period"]
            cost = settings["transaction_cost_rate"]

            if len(asset_returns) < training + holding:
                infeasible.append(value)
                continue

            key = (training, holding, cost)

            if key not in cache:
                cache[key] = walk_forward_backtest(
                    asset_returns,
                    strategy,
                    estimation_window=training,
                    holding_period=holding,
                    transaction_cost_rate=cost,
                )

            scenarios[value] = cache[key]

        if not scenarios:
            raise ValueError(f"no feasible scenarios for {parameter}")

        aligned = _align_strategy_returns(
            {str(value): result.net_returns for value, result in scenarios.items()}
        )

        start = aligned.index[0]
        end = aligned.index[-1]
        rows = {}

        for value, result in scenarios.items():
            metrics = performance_summary(aligned[str(value)])

            costs = result.transaction_costs.loc[
                (result.transaction_costs.index >= start)
                & (result.transaction_costs.index <= end)
            ]

            rows[value] = {
                **metrics,
                "total_transaction_cost": float(costs.sum()),
                "status": "OK",
            }

        for value in infeasible:
            rows[value] = {
                "cagr": np.nan,
                "volatility": np.nan,
                "sharpe": np.nan,
                "max_drawdown": np.nan,
                "total_transaction_cost": np.nan,
                "status": "Insufficient history",
            }

        table = pd.DataFrame.from_dict(rows, orient="index").reindex(options)

        table.index.name = parameter

        table.attrs = {
            "start_date": start,
            "end_date": end,
            "observations": len(aligned),
        }

        tables[parameter] = table

    return SensitivityResult(
        strategy=strategy,
        tables=tables,
    )
