"""Tests for out-of-sample strategy evaluation."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation import compare_strategies, performance_summary
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
