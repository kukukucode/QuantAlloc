"""Tests for walk-forward backtest input validation."""

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from src.backtest import (
    BacktestResult,
    _validate_backtest_inputs,
    walk_forward_backtest,
)


@pytest.fixture
def asset_returns() -> pd.DataFrame:
    index = pd.bdate_range("2022-01-03", periods=567)
    return pd.DataFrame(
        {
            "A": np.linspace(-0.01, 0.01, len(index)),
            "B": np.linspace(0.01, -0.01, len(index)),
        },
        index=index,
    )


def test_backtest_result_is_frozen() -> None:
    result = BacktestResult(
        gross_returns=pd.Series(dtype=float),
        net_returns=pd.Series(dtype=float),
        target_weights=pd.DataFrame(),
        pre_rebalance_weights=pd.DataFrame(),
        turnover=pd.Series(dtype=float),
        transaction_costs=pd.Series(dtype=float),
        periods=pd.DataFrame(),
    )

    with pytest.raises(FrozenInstanceError):
        result.gross_returns = pd.Series([0.01])


@pytest.mark.parametrize(
    "strategy",
    [
        "equal_weight",
        "minimum_variance",
        "maximum_sharpe",
        "risk_parity",
    ],
)
def test_validation_accepts_supported_strategies(
    asset_returns: pd.DataFrame,
    strategy: str,
) -> None:
    result = _validate_backtest_inputs(asset_returns, strategy)

    pd.testing.assert_frame_equal(result, asset_returns)
    assert result is not asset_returns


def test_validation_rejects_unknown_strategy(
    asset_returns: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="unknown strategy"):
        _validate_backtest_inputs(asset_returns, "unknown")


def test_validation_rejects_insufficient_data(
    asset_returns: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="at least 567 observations"):
        _validate_backtest_inputs(asset_returns.iloc[:-1], "equal_weight")


def test_validation_rejects_missing_values(
    asset_returns: pd.DataFrame,
) -> None:
    asset_returns.iloc[10, 0] = np.nan

    with pytest.raises(ValueError, match="missing values"):
        _validate_backtest_inputs(asset_returns, "minimum_variance")


def test_validation_rejects_unsorted_dates(
    asset_returns: pd.DataFrame,
) -> None:
    unsorted = asset_returns.iloc[::-1]

    with pytest.raises(ValueError, match="sorted in ascending order"):
        _validate_backtest_inputs(unsorted, "maximum_sharpe")


@pytest.mark.parametrize("cost_rate", [-0.001, 1.0, np.inf])
def test_validation_rejects_invalid_transaction_cost_rate(
    asset_returns: pd.DataFrame,
    cost_rate: float,
) -> None:
    with pytest.raises(ValueError, match="transaction_cost_rate"):
        _validate_backtest_inputs(
            asset_returns,
            "equal_weight",
            transaction_cost_rate=cost_rate,
        )


def test_first_oos_date_follows_training_end(
    asset_returns: pd.DataFrame,
) -> None:
    result = walk_forward_backtest(asset_returns, "equal_weight")
    first_period = result.periods.iloc[0]

    assert first_period["train_end"] == asset_returns.index[503]
    assert first_period["test_start"] == asset_returns.index[504]
    assert first_period["train_end"] < first_period["test_start"]


def test_windows_advance_by_holding_period() -> None:
    index = pd.bdate_range("2020-01-01", periods=630)
    returns = pd.DataFrame(
        {"A": 0.01, "B": -0.005},
        index=index,
    )

    result = walk_forward_backtest(returns, "equal_weight")

    assert len(result.periods) == 2
    assert result.periods.loc[1, "train_start"] == index[63]
    assert result.periods.loc[1, "test_start"] == index[567]
    assert len(result.returns) == 126


def test_incomplete_final_window_is_ignored() -> None:
    index = pd.bdate_range("2020-01-01", periods=580)
    returns = pd.DataFrame(
        {"A": 0.01, "B": -0.005},
        index=index,
    )

    result = walk_forward_backtest(returns, "equal_weight")

    assert len(result.periods) == 1
    assert len(result.returns) == 63
    assert result.returns.index[-1] == index[566]


def test_future_data_does_not_change_first_weights() -> None:
    generator = np.random.default_rng(42)
    index = pd.bdate_range("2020-01-01", periods=567)
    training = generator.normal(0.0005, 0.01, size=(504, 3))
    normal_future = generator.normal(0.0005, 0.01, size=(63, 3))
    extreme_future = np.full((63, 3), [0.50, -0.40, 0.25])

    dataset_a = pd.DataFrame(
        np.vstack([training, normal_future]),
        index=index,
        columns=["A", "B", "C"],
    )
    dataset_b = pd.DataFrame(
        np.vstack([training, extreme_future]),
        index=index,
        columns=["A", "B", "C"],
    )

    result_a = walk_forward_backtest(dataset_a, "maximum_sharpe")
    result_b = walk_forward_backtest(dataset_b, "maximum_sharpe")

    pd.testing.assert_series_equal(
        result_a.weights.iloc[0],
        result_b.weights.iloc[0],
    )


@pytest.mark.parametrize(
    "strategy",
    [
        "equal_weight",
        "minimum_variance",
        "maximum_sharpe",
        "risk_parity",
    ],
)
def test_backtest_weights_follow_constraints(strategy: str) -> None:
    generator = np.random.default_rng(7)
    index = pd.bdate_range("2020-01-01", periods=567)
    returns = pd.DataFrame(
        generator.normal(0.0005, 0.01, size=(567, 3)),
        index=index,
        columns=["A", "B", "C"],
    )

    result = walk_forward_backtest(returns, strategy)

    assert result.weights.sum(axis=1).to_numpy() == pytest.approx([1.0])
    assert (result.weights >= 0.0).all().all()
    assert (result.weights <= 1.0).all().all()


def test_oos_return_uses_fixed_period_weights() -> None:
    index = pd.bdate_range("2020-01-01", periods=567)
    returns = pd.DataFrame(
        {"A": 0.02, "B": 0.00},
        index=index,
    )

    result = walk_forward_backtest(returns, "equal_weight")

    asset_values = (1.0 + returns.iloc[504:567]).cumprod() * 0.5
    portfolio_values = asset_values.sum(axis=1)
    expected = portfolio_values.pct_change(fill_method=None)
    expected.iloc[0] = portfolio_values.iloc[0] - 1.0
    expected.name = "equal_weight"
    pd.testing.assert_series_equal(result.gross_returns, expected)


def test_first_turnover_is_zero(asset_returns: pd.DataFrame) -> None:
    result = walk_forward_backtest(asset_returns, "equal_weight")

    assert result.turnover.iloc[0] == pytest.approx(0.0)
    assert result.turnover.index.equals(result.weights.index)


def test_turnover_uses_drifted_pre_rebalance_weights() -> None:
    index = pd.bdate_range("2024-01-01", periods=6)
    returns = pd.DataFrame(
        {
            "A": [0.0, 0.0, 0.10, 0.0, 0.0, 0.0],
            "B": [0.0, 0.0, 0.00, 0.0, 0.0, 0.0],
        },
        index=index,
    )

    result = walk_forward_backtest(
        returns,
        "equal_weight",
        estimation_window=2,
        holding_period=2,
    )

    pre_rebalance_a = 0.5 * 1.10 / (0.5 * 1.10 + 0.5)
    expected_turnover = 2 * abs(0.5 - pre_rebalance_a)
    assert result.turnover.to_list() == pytest.approx(
        [0.0, expected_turnover]
    )
    assert result.pre_rebalance_weights.iloc[1, 0] == pytest.approx(
        pre_rebalance_a
    )


def test_transaction_cost_is_applied_to_first_return_after_rebalance() -> None:
    index = pd.bdate_range("2024-01-01", periods=6)
    returns = pd.DataFrame(
        {
            "A": [0.0, 0.0, 0.10, 0.0, 0.0, 0.0],
            "B": [0.0, 0.0, 0.00, 0.0, 0.0, 0.0],
        },
        index=index,
    )

    result = walk_forward_backtest(
        returns,
        "equal_weight",
        estimation_window=2,
        holding_period=2,
        transaction_cost_rate=0.01,
    )

    assert result.transaction_costs.iloc[0] == pytest.approx(0.0)
    assert result.transaction_costs.iloc[1] == pytest.approx(
        result.turnover.iloc[1] * 0.01
    )
    second_rebalance = result.transaction_costs.index[1]
    expected_net_return = (
        (1.0 + result.gross_returns.loc[second_rebalance])
        * (1.0 - result.transaction_costs.iloc[1])
        - 1.0
    )
    assert result.net_returns.loc[second_rebalance] == pytest.approx(
        expected_net_return
    )


def test_zero_cost_makes_gross_and_net_returns_equal(
    asset_returns: pd.DataFrame,
) -> None:
    result = walk_forward_backtest(
        asset_returns,
        "minimum_variance",
        transaction_cost_rate=0.0,
    )

    pd.testing.assert_series_equal(
        result.gross_returns,
        result.net_returns,
    )


def test_returns_and_weights_aliases_remain_compatible(
    asset_returns: pd.DataFrame,
) -> None:
    result = walk_forward_backtest(asset_returns, "equal_weight")

    assert result.returns is result.gross_returns
    assert result.weights is result.target_weights
