"""Tests for out-of-sample strategy evaluation."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    compare_rebalancing_frequencies,
    compare_strategies,
    compare_turnover,
    performance_summary,
    turnover_summary,
)
from src.metrics import (
    annualized_volatility,
    cagr,
    max_drawdown,
    sharpe_ratio,
)


def test_performance_summary_uses_existing_metrics() -> None:
    returns = pd.Series(
        [0.01, -0.02, 0.03],
        index=pd.to_datetime(
            ["2024-01-02", "2024-01-03", "2024-01-04"]
        ),
    )

    result = performance_summary(returns)

    assert set(result) == {
        "cagr",
        "volatility",
        "sharpe",
        "max_drawdown",
    }
    assert result == pytest.approx(
        {
            "cagr": cagr(returns),
            "volatility": annualized_volatility(returns),
            "sharpe": sharpe_ratio(returns),
            "max_drawdown": max_drawdown(returns),
        }
    )


def test_compare_strategies_preserves_strategy_names() -> None:
    index = pd.bdate_range("2024-01-01", periods=3)
    strategy_returns = {
        "Equal Weight": pd.Series([0.01, 0.00, -0.01], index=index),
        "TOPIX": pd.Series([0.00, 0.01, -0.02], index=index),
    }

    result = compare_strategies(strategy_returns)

    assert list(result.index) == ["Equal Weight", "TOPIX"]
    assert result.index.name == "Strategy"
    assert list(result.columns) == [
        "cagr",
        "volatility",
        "sharpe",
        "max_drawdown",
    ]


def test_compare_strategies_uses_only_common_dates() -> None:
    first_index = pd.bdate_range("2024-01-01", periods=4)
    second_index = pd.bdate_range("2024-01-02", periods=4)
    strategy_returns = {
        "Strategy A": pd.Series([0.50, 0.01, 0.02, 0.03], index=first_index),
        "Strategy B": pd.Series([0.02, 0.01, -0.01, -0.50], index=second_index),
    }

    result = compare_strategies(strategy_returns)

    common_index = first_index.intersection(second_index)
    expected_a = performance_summary(strategy_returns["Strategy A"].loc[common_index])
    expected_b = performance_summary(strategy_returns["Strategy B"].loc[common_index])
    assert result.loc["Strategy A"].to_dict() == pytest.approx(expected_a)
    assert result.loc["Strategy B"].to_dict() == pytest.approx(expected_b)
    assert result.attrs["start_date"] == common_index[0]
    assert result.attrs["end_date"] == common_index[-1]
    assert result.attrs["observations"] == len(common_index)


def test_outside_dates_do_not_affect_comparison() -> None:
    common_index = pd.bdate_range("2024-01-02", periods=3)
    extended_index = common_index.insert(0, pd.Timestamp("2024-01-01"))
    strategy_returns = {
        "Strategy A": pd.Series([0.01, 0.02, -0.01], index=common_index),
        "Strategy B": pd.Series([10.0, 0.01, 0.02, -0.01], index=extended_index),
    }

    result = compare_strategies(strategy_returns)

    assert result.loc["Strategy A"].to_dict() == pytest.approx(
        result.loc["Strategy B"].to_dict()
    )


def test_compare_strategies_rejects_nan_in_common_period() -> None:
    index = pd.bdate_range("2024-01-01", periods=3)
    strategy_returns = {
        "Strategy A": pd.Series([0.01, np.nan, -0.01], index=index),
        "Strategy B": pd.Series([0.00, 0.01, -0.02], index=index),
    }

    with pytest.raises(ValueError, match="must not contain NaN"):
        compare_strategies(strategy_returns)


def test_compare_strategies_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        compare_strategies({})


def test_compare_strategies_rejects_non_overlapping_dates() -> None:
    strategy_returns = {
        "Strategy A": pd.Series(
            [0.01, 0.02],
            index=pd.bdate_range("2024-01-01", periods=2),
        ),
        "Strategy B": pd.Series(
            [0.01, 0.02],
            index=pd.bdate_range("2024-02-01", periods=2),
        ),
    }

    with pytest.raises(ValueError, match="share at least two"):
        compare_strategies(strategy_returns)


def test_turnover_summary() -> None:
    turnover = pd.Series([0.0, 0.20, 0.40])

    result = turnover_summary(turnover)

    assert result == pytest.approx(
        {
            "average_turnover": 0.20,
            "total_turnover": 0.60,
            "maximum_turnover": 0.40,
        }
    )


def test_compare_turnover_preserves_strategy_names() -> None:
    strategy_turnover = {
        "Equal Weight": pd.Series([0.0, 0.05]),
        "Maximum Sharpe": pd.Series([0.0, 0.40]),
    }

    result = compare_turnover(strategy_turnover)

    assert list(result.index) == ["Equal Weight", "Maximum Sharpe"]
    assert list(result.columns) == [
        "average_turnover",
        "total_turnover",
        "maximum_turnover",
    ]


@pytest.mark.parametrize(
    "turnover",
    [pd.Series([0.0, np.nan]), pd.Series([0.0, -0.1])],
)
def test_turnover_summary_rejects_invalid_values(
    turnover: pd.Series,
) -> None:
    with pytest.raises(ValueError):
        turnover_summary(turnover)


def test_frequency_comparison_uses_common_oos_period() -> None:
    generator = np.random.default_rng(123)
    index = pd.bdate_range("2023-01-02", periods=20)
    returns = pd.DataFrame(
        generator.normal(0.001, 0.01, size=(20, 3)),
        index=index,
        columns=["A", "B", "C"],
    )

    result = compare_rebalancing_frequencies(
        returns,
        strategies=("equal_weight",),
        holding_periods=(2, 4),
        estimation_window=4,
        transaction_cost_rate=0.001,
    )

    assert list(result.index) == [("equal_weight", 2), ("equal_weight", 4)]
    assert result.attrs["start_date"] == index[4]
    assert result.attrs["end_date"] == index[19]
    assert result.attrs["observations"] == 16
    assert set(result.columns) == {
        "rebalances",
        "net_cagr",
        "net_volatility",
        "net_sharpe",
        "net_max_drawdown",
        "average_turnover",
        "total_turnover",
        "maximum_turnover",
        "total_transaction_cost",
    }


def test_more_frequent_rebalancing_records_more_rebalances() -> None:
    index = pd.bdate_range("2023-01-02", periods=20)
    returns = pd.DataFrame(
        {
            "A": np.tile([0.02, -0.01], 10),
            "B": np.tile([-0.01, 0.02], 10),
        },
        index=index,
    )

    result = compare_rebalancing_frequencies(
        returns,
        strategies=("equal_weight",),
        holding_periods=(2, 4),
        estimation_window=4,
        transaction_cost_rate=0.001,
    )

    frequent = result.loc[("equal_weight", 2)]
    infrequent = result.loc[("equal_weight", 4)]
    assert frequent["rebalances"] > infrequent["rebalances"]
    assert frequent["total_transaction_cost"] >= 0.0
    assert infrequent["total_transaction_cost"] >= 0.0


def test_frequency_comparison_rejects_empty_configuration() -> None:
    returns = pd.DataFrame(
        {"A": [0.01, 0.02]},
        index=pd.bdate_range("2024-01-01", periods=2),
    )

    with pytest.raises(ValueError, match="strategies must not be empty"):
        compare_rebalancing_frequencies(returns, strategies=())


def test_frequency_comparison_supports_risk_parity() -> None:
    generator = np.random.default_rng(456)
    index = pd.bdate_range("2023-01-02", periods=20)
    returns = pd.DataFrame(
        generator.normal(0.001, 0.01, size=(20, 3)),
        index=index,
        columns=["A", "B", "C"],
    )

    result = compare_rebalancing_frequencies(
        returns,
        strategies=("risk_parity",),
        holding_periods=(2,),
        estimation_window=4,
        transaction_cost_rate=0.001,
    )

    assert list(result.index) == [("risk_parity", 2)]
    assert result.loc[("risk_parity", 2), "rebalances"] == 8
    assert result.loc[("risk_parity", 2), "total_transaction_cost"] >= 0.0
