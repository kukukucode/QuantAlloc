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
        returns=pd.Series(dtype=float),
        weights=pd.DataFrame(),
        periods=pd.DataFrame(),
    )

    with pytest.raises(FrozenInstanceError):
        result.returns = pd.Series([0.01])


@pytest.mark.parametrize(
    "strategy",
    ["equal_weight", "minimum_variance", "maximum_sharpe"],
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
    ["equal_weight", "minimum_variance", "maximum_sharpe"],
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

    expected = pd.Series(
        0.01,
        index=index[504:567],
        name="equal_weight",
    )
    pd.testing.assert_series_equal(result.returns, expected)
