"""Tests for rolling annualized risk metrics."""

import math

import numpy as np
import pandas as pd
import pytest

from src.rolling import rolling_risk_metrics


def test_rolling_metrics_match_manual_calculation() -> None:
    returns = pd.DataFrame(
        {
            "A": [0.02, -0.01, 0.015, 0.005],
            "B": [0.0, 0.0, 0.0, 0.0],
        },
        index=pd.bdate_range("2024-01-01", periods=4),
    )

    volatility, sharpe = rolling_risk_metrics(returns, window=3)

    assert volatility.iloc[:2].isna().all().all()
    assert sharpe.iloc[:2].isna().all().all()

    first_window = returns["A"].iloc[:3]
    expected_vol = first_window.std(ddof=1) * math.sqrt(252)
    expected_sharpe = first_window.mean() * 252 / expected_vol

    assert volatility.iloc[2]["A"] == pytest.approx(expected_vol)
    assert sharpe.iloc[2]["A"] == pytest.approx(expected_sharpe)

    assert (volatility["B"].iloc[2:] == 0).all()
    assert sharpe["B"].isna().all()


def test_insufficient_history_returns_nan() -> None:
    returns = pd.DataFrame(
        {"A": [0.01, -0.02, 0.03]},
        index=pd.bdate_range("2024-01-01", periods=3),
    )

    volatility, sharpe = rolling_risk_metrics(returns, window=63)

    assert volatility.isna().all().all()
    assert sharpe.isna().all().all()


@pytest.mark.parametrize("window", [0, 1, -1, 2.5, True])
def test_rejects_invalid_window(window) -> None:
    returns = pd.DataFrame(
        {"A": [0.01, -0.02]},
        index=pd.bdate_range("2024-01-01", periods=2),
    )

    with pytest.raises(ValueError, match="window"):
        rolling_risk_metrics(returns, window)


@pytest.mark.parametrize("periods", [0, -1, 252.5, True])
def test_rejects_invalid_annualization(periods) -> None:
    returns = pd.DataFrame(
        {"A": [0.01, -0.02]},
        index=pd.bdate_range("2024-01-01", periods=2),
    )

    with pytest.raises(ValueError, match="periods_per_year"):
        rolling_risk_metrics(returns, window=2, periods_per_year=periods)


@pytest.mark.parametrize(
    "values",
    [[np.nan, 0.02], [np.inf, 0.02], [-1.5, 0.02]],
)
def test_rejects_invalid_returns(values) -> None:
    returns = pd.DataFrame(
        {"A": values},
        index=pd.bdate_range("2024-01-01", periods=2),
    )

    with pytest.raises(ValueError):
        rolling_risk_metrics(returns, window=2)


def test_rejects_non_datetime_index() -> None:
    returns = pd.DataFrame({"A": [0.01, -0.02]})

    with pytest.raises(TypeError, match="DatetimeIndex"):
        rolling_risk_metrics(returns, window=2)
