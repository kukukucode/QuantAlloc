"""Verify parameter sensitivity and common OOS evaluation."""

import pandas as pd
import pytest

from src.backtest import walk_forward_backtest
from src.evaluation import _align_strategy_returns, performance_summary
from src.portfolio import calculate_asset_returns
from src.sensitivity import run_parameter_sensitivity


@pytest.fixture
def asset_returns(sample_market_prices):
    return calculate_asset_returns(sample_market_prices[0])


def test_all_sweeps_have_expected_scenarios(asset_returns) -> None:
    result = run_parameter_sensitivity(asset_returns, "equal_weight")

    assert result.strategy == "equal_weight"
    assert list(result.tables) == [
        "estimation_window",
        "holding_period",
        "transaction_cost_rate",
    ]

    assert result.tables["estimation_window"].index.tolist() == [252, 378, 504]
    assert result.tables["holding_period"].index.tolist() == [21, 63, 126, 252]
    assert result.tables["transaction_cost_rate"].index.tolist() == [0.0, 0.001, 0.003]


def test_insufficient_history_is_explicit(asset_returns) -> None:
    result = run_parameter_sensitivity(asset_returns, "equal_weight")

    holding = result.tables["holding_period"]

    assert holding.loc[252, "status"] == "Insufficient history"
    assert pd.isna(holding.loc[252, "cagr"])
    assert (holding.loc[[21, 63, 126], "status"] == "OK").all()


def test_training_comparison_uses_identical_oos_dates(
    asset_returns,
) -> None:
    result = run_parameter_sensitivity(asset_returns, "equal_weight")
    table = result.tables["estimation_window"]

    backtests = {
        window: walk_forward_backtest(
            asset_returns,
            "equal_weight",
            estimation_window=window,
            holding_period=63,
            transaction_cost_rate=0.001,
        )
        for window in (252, 378, 504)
    }

    aligned = _align_strategy_returns(
        {str(window): backtest.net_returns for window, backtest in backtests.items()}
    )

    assert table.attrs["start_date"] == aligned.index[0]
    assert table.attrs["end_date"] == aligned.index[-1]
    assert table.attrs["observations"] == len(aligned)

    for window in backtests:
        expected = performance_summary(aligned[str(window)])

        for metric, value in expected.items():
            assert table.loc[window, metric] == pytest.approx(value, nan_ok=True)


def test_cost_comparison_uses_same_dates_and_shows_cost(
    asset_returns,
) -> None:
    result = run_parameter_sensitivity(asset_returns, "equal_weight")
    table = result.tables["transaction_cost_rate"]

    assert (table["status"] == "OK").all()
    assert table.loc[0.0, "total_transaction_cost"] == 0.0
    assert (
        table.loc[0.003, "total_transaction_cost"]
        >= table.loc[0.001, "total_transaction_cost"]
    )

    # Higher costs cannot improve the realized net CAGR when
    # all other settings and rebalance decisions are identical.
    assert table.loc[0.0, "cagr"] >= table.loc[0.001, "cagr"]
    assert table.loc[0.001, "cagr"] >= table.loc[0.003, "cagr"]


def test_rejects_unknown_strategy(asset_returns) -> None:
    with pytest.raises(ValueError, match="unsupported strategy"):
        run_parameter_sensitivity(asset_returns, "unknown")


@pytest.mark.parametrize(
    "options",
    [(), (0,), (252, 252), (True,), (252.5,)],
)
def test_rejects_invalid_training_options(asset_returns, options) -> None:
    with pytest.raises(ValueError):
        run_parameter_sensitivity(
            asset_returns,
            "equal_weight",
            estimation_windows=options,
        )


@pytest.mark.parametrize(
    "options",
    [(), (0,), (21, 21), (True,), (-1,)],
)
def test_rejects_invalid_holding_options(asset_returns, options) -> None:
    with pytest.raises(ValueError):
        run_parameter_sensitivity(
            asset_returns,
            "equal_weight",
            holding_periods=options,
        )


@pytest.mark.parametrize(
    "options",
    [(), (-0.1,), (1.0,), (0.001, 0.001), (True,)],
)
def test_rejects_invalid_cost_options(asset_returns, options) -> None:
    with pytest.raises(ValueError):
        run_parameter_sensitivity(
            asset_returns,
            "equal_weight",
            transaction_cost_rates=options,
        )


def test_rejects_unavailable_sweep(asset_returns) -> None:
    with pytest.raises(ValueError, match="no feasible scenarios"):
        run_parameter_sensitivity(
            asset_returns,
            "equal_weight",
            estimation_windows=(900,),
        )
