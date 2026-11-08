"""One-factor-at-a-time walk-forward parameter sensitivity."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from multiprocessing import get_context

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.backtest import BacktestResult, walk_forward_backtest
from src.dashboard_settings import DashboardSettings
from src.evaluation import _align_strategy_returns, performance_summary
from src.sample_config import (
    ANALYSIS_STRATEGY_LABELS,
    ESTIMATION_WINDOW,
    HOLDING_PERIOD,
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


def _reprice_backtest(
    result: BacktestResult,
    transaction_cost_rate: float,
) -> BacktestResult:
    """Apply a new cost to fixed gross returns and rebalance turnover.

    The current backtester does not use transaction costs to select weights,
    and holding-period weight drift is based on gross asset returns.
    Repricing is not valid for cost-aware optimizers or cash-drag models.
    """
    transaction_costs = result.turnover * transaction_cost_rate
    transaction_costs.name = result.transaction_costs.name
    unaffordable = transaction_costs >= 1.0
    if unaffordable.any():
        rebalance = transaction_costs.index[unaffordable][0]
        raise ValueError(
            "transaction costs must be less than portfolio value "
            f"at rebalance {rebalance}"
        )

    net_returns = result.gross_returns.copy()
    rebalance_dates = transaction_costs.index
    gross_first_days = net_returns.loc[rebalance_dates].to_numpy()
    net_returns.loc[rebalance_dates] = (1.0 + gross_first_days) * (
        1.0 - transaction_costs.to_numpy()
    ) - 1.0
    net_returns.name = result.net_returns.name
    return replace(
        result,
        net_returns=net_returns,
        transaction_costs=transaction_costs,
    )


# Initialized once per worker to avoid serializing prices for every scenario.
_WORKER_RETURNS: pd.DataFrame | None = None
_WORKER_STRATEGY: str | None = None
_WORKER_BLAS_LIMIT = None


def _initialize_sensitivity_worker(asset_returns: pd.DataFrame, strategy: str) -> None:
    global _WORKER_RETURNS, _WORKER_STRATEGY, _WORKER_BLAS_LIMIT
    _WORKER_RETURNS = asset_returns
    _WORKER_STRATEGY = strategy
    # Prevent several BLAS threads in each process from oversubscribing CPU.
    _WORKER_BLAS_LIMIT = threadpool_limits(limits=1)


def _run_sensitivity_worker(
    key: tuple[int, int],
) -> tuple[tuple[int, int], BacktestResult]:
    if _WORKER_RETURNS is None or _WORKER_STRATEGY is None:
        raise RuntimeError("sensitivity worker is not initialized")
    estimation_window, holding_period = key
    result = walk_forward_backtest(
        _WORKER_RETURNS,
        _WORKER_STRATEGY,
        estimation_window=estimation_window,
        holding_period=holding_period,
        transaction_cost_rate=0.0,
    )
    return key, result


def run_parameter_sensitivity(
    asset_returns: pd.DataFrame,
    strategy: str,
    *,
    estimation_window: int = ESTIMATION_WINDOW,
    holding_period: int = HOLDING_PERIOD,
    transaction_cost_rate: float = TRANSACTION_COST_RATE,
    estimation_windows: tuple[int, ...] = (252, 378, 504),
    holding_periods: tuple[int, ...] = (21, 63, 126, 252),
    transaction_cost_rates: tuple[float, ...] = (0.0, 0.001, 0.003),
    max_workers: int = 1,
) -> SensitivityResult:
    """Change one parameter at a time, aligning OOS dates within each sweep.

    Infeasible scenarios are shown explicitly instead of silently replacing
    them with different parameter values. Identical configurations are cached.
    """
    if strategy not in ANALYSIS_STRATEGY_LABELS:
        raise ValueError(f"unsupported strategy: {strategy}")
    if (
        isinstance(max_workers, bool)
        or not isinstance(max_workers, int)
        or not 1 <= max_workers <= 4
    ):
        raise ValueError("max_workers must be an integer between 1 and 4")
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

    config = DashboardSettings(
        estimation_window=estimation_window,
        holding_period=holding_period,
        transaction_cost_rate=transaction_cost_rate,
    )
    baseline = {
        "estimation_window": config.estimation_window,
        "holding_period": config.holding_period,
        "transaction_cost_rate": config.transaction_cost_rate,
    }

    # Only independent (estimation_window, holding_period) pairs are scheduled.
    # Different transaction costs reuse the same cost-free backtest.
    keys: list[tuple[int, int]] = []
    for parameter, options in sweeps.items():
        feasible = False
        for value in options:
            settings = baseline.copy()
            settings[parameter] = value
            key = (settings["estimation_window"], settings["holding_period"])
            if len(asset_returns) >= sum(key):
                feasible = True
                if key not in keys:
                    keys.append(key)
        if not feasible:
            raise ValueError(f"no feasible scenarios for {parameter}")

    if max_workers > 1 and len(keys) > 1:
        # spawn works across platforms and prevents unsafe fork of BLAS/Rust.
        # The input frame is copied once per process; keep worker count small.
        with ProcessPoolExecutor(
            max_workers=min(max_workers, len(keys)),
            mp_context=get_context("spawn"),
            initializer=_initialize_sensitivity_worker,
            initargs=(asset_returns, strategy),
        ) as pool:
            gross_cache = dict(pool.map(_run_sensitivity_worker, keys))
    else:
        gross_cache = {
            (training, holding): walk_forward_backtest(
                asset_returns,
                strategy,
                estimation_window=training,
                holding_period=holding,
                transaction_cost_rate=0.0,
            )
            for training, holding in keys
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
                gross_key = (training, holding)
                cache[key] = _reprice_backtest(gross_cache[gross_key], cost)

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
