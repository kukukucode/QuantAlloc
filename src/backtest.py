"""Walk-forward backtesting primitives for QuantAlloc."""

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from src.diversification import covariance_matrix
from src.optimization import (
    expected_returns,
    maximum_sharpe_weights,
    minimum_variance_weights,
)


VALID_STRATEGIES = frozenset(
    {
        "equal_weight",
        "minimum_variance",
        "maximum_sharpe",
    }
)


@dataclass(frozen=True)
class BacktestResult:
    """Out-of-sample returns, rebalance weights, and period audit data."""

    returns: pd.Series
    weights: pd.DataFrame
    periods: pd.DataFrame


def _validate_backtest_inputs(
    asset_returns: pd.DataFrame,
    strategy: str,
    estimation_window: int = 504,
    holding_period: int = 63,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """Validate and copy inputs for a walk-forward backtest."""
    if strategy not in VALID_STRATEGIES:
        valid = ", ".join(sorted(VALID_STRATEGIES))
        raise ValueError(f"unknown strategy: {strategy}; expected one of {valid}")

    if isinstance(estimation_window, bool) or not isinstance(
        estimation_window,
        int,
    ):
        raise ValueError("estimation_window must be an integer of at least 2")
    if estimation_window < 2:
        raise ValueError("estimation_window must be an integer of at least 2")

    if isinstance(holding_period, bool) or not isinstance(holding_period, int):
        raise ValueError("holding_period must be a positive integer")
    if holding_period <= 0:
        raise ValueError("holding_period must be a positive integer")

    if not math.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be finite")

    if not isinstance(asset_returns, pd.DataFrame) or asset_returns.empty:
        raise ValueError("asset_returns must be a non-empty DataFrame")
    if not isinstance(asset_returns.index, pd.DatetimeIndex):
        raise TypeError("asset_returns must use a DatetimeIndex")
    if not asset_returns.index.is_monotonic_increasing:
        raise ValueError("asset_returns index must be sorted in ascending order")
    if asset_returns.index.has_duplicates:
        raise ValueError("asset_returns index must be unique")
    if asset_returns.columns.has_duplicates:
        raise ValueError("asset_returns columns must be unique")

    minimum_observations = estimation_window + holding_period
    if len(asset_returns) < minimum_observations:
        raise ValueError(
            "asset_returns must contain at least "
            f"{minimum_observations} observations"
        )

    try:
        validated = asset_returns.apply(pd.to_numeric, errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("asset_returns must be numeric") from exc

    if validated.isna().any().any():
        raise ValueError("asset_returns must not contain missing values")
    if not np.isfinite(validated.to_numpy()).all():
        raise ValueError("asset_returns must contain only finite values")

    return validated.copy()


def _calculate_weights(
    training_returns: pd.DataFrame,
    strategy: str,
    risk_free_rate: float,
) -> pd.Series:
    """Calculate weights using training data only."""
    if strategy == "equal_weight":
        asset_count = len(training_returns.columns)
        return pd.Series(
            1.0 / asset_count,
            index=training_returns.columns,
            name="equal_weight",
        )

    covariance = covariance_matrix(training_returns)
    if strategy == "minimum_variance":
        return minimum_variance_weights(covariance)

    historical_returns = expected_returns(training_returns)
    return maximum_sharpe_weights(
        historical_returns,
        covariance,
        risk_free_rate,
    )


def walk_forward_backtest(
    asset_returns: pd.DataFrame,
    strategy: str,
    estimation_window: int = 504,
    holding_period: int = 63,
    risk_free_rate: float = 0.0,
) -> BacktestResult:
    """Run a rolling, fixed-holding-period out-of-sample backtest."""
    validated = _validate_backtest_inputs(
        asset_returns,
        strategy,
        estimation_window,
        holding_period,
        risk_free_rate,
    )

    oos_returns: list[pd.Series] = []
    weight_rows: list[pd.Series] = []
    rebalance_dates: list[pd.Timestamp] = []
    period_rows: list[dict[str, pd.Timestamp]] = []

    final_test_start = len(validated) - holding_period
    for test_start_position in range(
        estimation_window,
        final_test_start + 1,
        holding_period,
    ):
        train_start_position = test_start_position - estimation_window
        test_end_position = test_start_position + holding_period

        training = validated.iloc[train_start_position:test_start_position]
        test = validated.iloc[test_start_position:test_end_position]
        weights = _calculate_weights(
            training,
            strategy,
            risk_free_rate,
        ).reindex(validated.columns)

        period_return = test.dot(weights)
        period_return.name = strategy
        oos_returns.append(period_return)

        rebalance_date = test.index[0]
        weight_rows.append(weights)
        rebalance_dates.append(rebalance_date)
        period_rows.append(
            {
                "rebalance": rebalance_date,
                "train_start": training.index[0],
                "train_end": training.index[-1],
                "test_start": test.index[0],
                "test_end": test.index[-1],
            }
        )

    combined_returns = pd.concat(oos_returns)
    combined_returns.name = strategy

    weight_history = pd.DataFrame(
        weight_rows,
        index=pd.DatetimeIndex(rebalance_dates, name="rebalance"),
    )
    weight_history = weight_history.reindex(columns=validated.columns)

    periods = pd.DataFrame(
        period_rows,
        columns=[
            "rebalance",
            "train_start",
            "train_end",
            "test_start",
            "test_end",
        ],
    )

    return BacktestResult(
        returns=combined_returns,
        weights=weight_history,
        periods=periods,
    )
