"""Validate interactive settings and their effect on analysis."""

import pandas as pd
import pytest

from src.backtest import walk_forward_backtest
from src.dashboard import (
    run_sample_analysis,
    run_sample_sensitivity,
    run_strategy_comparison,
)
from src.dashboard_settings import DashboardSettings
from src.portfolio import calculate_asset_returns
from src.sample_config import (
    ESTIMATION_WINDOW,
    HOLDING_PERIOD,
    TRANSACTION_COST_RATE,
)


def test_defaults_match_existing_configuration() -> None:
    settings = DashboardSettings()

    assert settings.estimation_window == ESTIMATION_WINDOW
    assert settings.holding_period == HOLDING_PERIOD
    assert settings.transaction_cost_rate == TRANSACTION_COST_RATE


@pytest.mark.parametrize(
    "kwargs",
    [
        {"estimation_window": True},
        {"estimation_window": 0},
        {"estimation_window": 252.5},
        {"holding_period": False},
        {"holding_period": 0},
        {"transaction_cost_rate": -0.1},
        {"transaction_cost_rate": 1.0},
        {"transaction_cost_rate": float("nan")},
        {"transaction_cost_rate": True},
    ],
)
def test_invalid_settings_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        DashboardSettings(**kwargs)


def test_single_analysis_uses_selected_settings(
    sample_market_prices,
) -> None:
    asset_prices, benchmark_prices = sample_market_prices

    settings = DashboardSettings(
        estimation_window=252,
        holding_period=126,
        transaction_cost_rate=0.003,
    )

    actual = run_sample_analysis(
        asset_prices,
        benchmark_prices,
        "equal_weight",
        settings=settings,
    )

    expected = walk_forward_backtest(
        calculate_asset_returns(asset_prices),
        "equal_weight",
        estimation_window=252,
        holding_period=126,
        transaction_cost_rate=0.003,
    )

    pd.testing.assert_series_equal(
        actual.backtest.net_returns,
        expected.net_returns,
    )


def test_comparison_uses_selected_settings(
    sample_market_prices,
) -> None:
    asset_prices, benchmark_prices = sample_market_prices

    settings = DashboardSettings(
        estimation_window=378,
        holding_period=126,
        transaction_cost_rate=0.0,
    )

    result = run_strategy_comparison(
        asset_prices,
        benchmark_prices,
        settings=settings,
    )

    for backtest in result.backtests.values():
        assert len(backtest.periods) >= 1
        assert (backtest.transaction_costs == 0).all()


def test_sensitivity_uses_selected_baseline(
    sample_market_prices,
) -> None:
    asset_prices, _ = sample_market_prices

    settings = DashboardSettings(
        estimation_window=378,
        holding_period=126,
        transaction_cost_rate=0.003,
    )

    result = run_sample_sensitivity(
        asset_prices,
        "equal_weight",
        settings=settings,
    )

    cost_table = result.tables["transaction_cost_rate"]

    assert (cost_table["status"] == "OK").all()
    assert cost_table.loc[0.0, "total_transaction_cost"] == 0.0
    assert cost_table.loc[0.003, "total_transaction_cost"] >= 0.0
