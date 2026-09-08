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
    """Gross/net OOS results and rebalance audit data."""

    gross_returns: pd.Series
    net_returns: pd.Series
    target_weights: pd.DataFrame
    pre_rebalance_weights: pd.DataFrame
    turnover: pd.Series
    transaction_costs: pd.Series
    periods: pd.DataFrame

    @property
    def returns(self) -> pd.Series:
        """Backward-compatible alias for gross returns."""
        return self.gross_returns

    @property
    def weights(self) -> pd.DataFrame:
        """Backward-compatible alias for target weights."""
        return self.target_weights


def _validate_backtest_inputs(
    asset_returns: pd.DataFrame,
    strategy: str,
    estimation_window: int = 504,
    holding_period: int = 63,
    risk_free_rate: float = 0.0,
    transaction_cost_rate: float = 0.001,
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
    if not math.isfinite(transaction_cost_rate):
        raise ValueError("transaction_cost_rate must be finite")
    if not 0.0 <= transaction_cost_rate < 1.0:
        raise ValueError(
            "transaction_cost_rate must be between 0.0 and 1.0"
        )

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


def _holding_period_returns(
    target_weights: pd.Series,
    holding_returns: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Simulate buy-and-hold returns and ending drifted weights."""
    asset_values = (1.0 + holding_returns).cumprod().mul(
        target_weights,
        axis="columns",
    )
    portfolio_values = asset_values.sum(axis=1)
    if (portfolio_values <= 0.0).any():
        raise ValueError("portfolio value must remain positive")

    gross_returns = portfolio_values.pct_change(fill_method=None)
    gross_returns.iloc[0] = portfolio_values.iloc[0] - 1.0
    gross_returns.name = "gross_return"

    ending_weights = asset_values.iloc[-1] / portfolio_values.iloc[-1]
    ending_weights.name = "pre_rebalance_weight"
    return gross_returns, ending_weights


def walk_forward_backtest(
    asset_returns: pd.DataFrame,
    strategy: str,
    estimation_window: int = 504,
    holding_period: int = 63,
    risk_free_rate: float = 0.0,
    transaction_cost_rate: float = 0.001,
) -> BacktestResult:
    """Run a rolling, fixed-holding-period out-of-sample backtest."""
    validated = _validate_backtest_inputs(
        asset_returns,
        strategy,
        estimation_window,
        holding_period,
        risk_free_rate,
        transaction_cost_rate,
    )

    gross_return_periods: list[pd.Series] = []
    net_return_periods: list[pd.Series] = []
    weight_rows: list[pd.Series] = []
    pre_rebalance_rows: list[pd.Series] = []
    rebalance_dates: list[pd.Timestamp] = []
    period_rows: list[dict[str, pd.Timestamp]] = []
    turnover_values: list[float] = []
    transaction_cost_values: list[float] = []
    previous_ending_weights: pd.Series | None = None

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

        if previous_ending_weights is None:
            before_rebalance = weights.copy()
            before_rebalance.name = "pre_rebalance_weight"
            turnover = 0.0
        else:
            before_rebalance = previous_ending_weights
            turnover = float((weights - before_rebalance).abs().sum())
        transaction_cost = turnover * transaction_cost_rate

        gross_returns, ending_weights = _holding_period_returns(weights, test)
        gross_returns.name = strategy
        net_returns = gross_returns.copy()
        net_returns.iloc[0] = (
            (1.0 + gross_returns.iloc[0]) * (1.0 - transaction_cost) - 1.0
        )
        net_returns.name = strategy
        gross_return_periods.append(gross_returns)
        net_return_periods.append(net_returns)

        rebalance_date = test.index[0]
        weight_rows.append(weights)
        pre_rebalance_rows.append(before_rebalance)
        rebalance_dates.append(rebalance_date)
        turnover_values.append(turnover)
        transaction_cost_values.append(transaction_cost)
        period_rows.append(
            {
                "rebalance": rebalance_date,
                "train_start": training.index[0],
                "train_end": training.index[-1],
                "test_start": test.index[0],
                "test_end": test.index[-1],
            }
        )
        previous_ending_weights = ending_weights

    combined_gross_returns = pd.concat(gross_return_periods)
    combined_gross_returns.name = strategy
    combined_net_returns = pd.concat(net_return_periods)
    combined_net_returns.name = strategy

    weight_history = pd.DataFrame(
        weight_rows,
        index=pd.DatetimeIndex(rebalance_dates, name="rebalance"),
    )
    weight_history = weight_history.reindex(columns=validated.columns)
    pre_rebalance_history = pd.DataFrame(
        pre_rebalance_rows,
        index=pd.DatetimeIndex(rebalance_dates, name="rebalance"),
    )
    pre_rebalance_history = pre_rebalance_history.reindex(
        columns=validated.columns
    )

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
    turnover_history = pd.Series(
        turnover_values,
        index=pd.DatetimeIndex(rebalance_dates, name="rebalance"),
        name="turnover",
    )
    transaction_cost_history = pd.Series(
        transaction_cost_values,
        index=pd.DatetimeIndex(rebalance_dates, name="rebalance"),
        name="transaction_cost",
    )

    return BacktestResult(
        gross_returns=combined_gross_returns,
        net_returns=combined_net_returns,
        target_weights=weight_history,
        pre_rebalance_weights=pre_rebalance_history,
        turnover=turnover_history,
        transaction_costs=transaction_cost_history,
        periods=periods,
    )
