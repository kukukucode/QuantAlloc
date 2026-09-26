"""Tests for portfolio validation and return calculations."""

import pandas as pd
import pytest

from src.portfolio import (
    calculate_asset_returns,
    calculate_portfolio_returns,
    validate_weights,
)


def test_validate_weights_returns_series() -> None:
    weights = {"7203.T": 0.4, "6758.T": 0.3, "1306.T": 0.3}

    result = validate_weights(weights, list(weights))

    pd.testing.assert_series_equal(
        result,
        pd.Series(weights, dtype=float, name="weight"),
    )


def test_validate_weights_rejects_sum_other_than_one() -> None:
    with pytest.raises(ValueError, match="sum to 1.0"):
        validate_weights({"A": 0.6, "B": 0.3}, ["A", "B"])


def test_validate_weights_rejects_negative_weight() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        validate_weights({"A": 1.1, "B": -0.1}, ["A", "B"])


def test_validate_weights_rejects_unknown_ticker() -> None:
    with pytest.raises(ValueError, match="without price data"):
        validate_weights({"A": 0.5, "B": 0.5}, ["A"])


def test_calculate_asset_returns() -> None:
    prices = pd.DataFrame(
        {
            "A": [100.0, 110.0, 121.0],
            "B": [200.0, 200.0, 220.0],
        },
        index=pd.to_datetime(["2025-01-06", "2025-01-07", "2025-01-08"]),
    )

    result = calculate_asset_returns(prices)

    expected = pd.DataFrame(
        {
            "A": [0.1, 0.1],
            "B": [0.0, 0.1],
        },
        index=pd.to_datetime(["2025-01-07", "2025-01-08"]),
    )
    pd.testing.assert_frame_equal(result, expected)


def test_calculate_portfolio_returns() -> None:
    returns = pd.DataFrame(
        {"A": [0.10], "B": [0.00]},
        index=pd.to_datetime(["2025-01-06"]),
    )
    weights = pd.Series({"A": 0.5, "B": 0.5})

    result = calculate_portfolio_returns(returns, weights)

    expected = pd.Series(
        [0.05],
        index=returns.index,
        name="portfolio_return",
    )
    pd.testing.assert_series_equal(result, expected)


def test_validate_weights_preserves_series_labels() -> None:
    weights = pd.Series([0.4, 0.6], index=["B", "A"])
    result = validate_weights(weights, ["A", "B"])
    pd.testing.assert_series_equal(result, weights.rename("weight"))


def test_duplicate_weight_assets_are_rejected_before_conversion() -> None:
    weights = pd.Series([0.2, 0.5, 0.5], index=["A", "A", "B"])
    returns = pd.DataFrame({"A": [0.1], "B": [0.0]})
    with pytest.raises(ValueError, match="index must be unique"):
        validate_weights(weights, ["A", "B"])
    with pytest.raises(ValueError, match="index must be unique"):
        calculate_portfolio_returns(returns, weights)


def test_portfolio_returns_reject_losses_below_minus_one() -> None:
    returns = pd.DataFrame({"A": [-1.5], "B": [0.0]})
    with pytest.raises(ValueError, match="less than -100%"):
        calculate_portfolio_returns(returns, pd.Series({"A": 0.5, "B": 0.5}))
