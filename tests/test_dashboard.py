"""Check dashboard wiring against the existing numeric engine."""

import pandas as pd
import pytest

from src.backtest import walk_forward_backtest
from src.dashboard import run_sample_analysis
from src.evaluation import performance_summary
from src.portfolio import calculate_asset_returns
from src.sample_config import (
    ESTIMATION_WINDOW, HOLDING_PERIOD, STRATEGY_LABELS, TRANSACTION_COST_RATE,
)


@pytest.mark.parametrize("strategy", list(STRATEGY_LABELS))
def test_dashboard_reuses_selected_strategy_net_backtest(sample_market_prices, strategy) -> None:
    asset_prices, benchmark_prices = sample_market_prices
    originals = (asset_prices.copy(deep=True), benchmark_prices.copy(deep=True))
    result = run_sample_analysis(asset_prices, benchmark_prices, strategy)
    expected = walk_forward_backtest(
        calculate_asset_returns(asset_prices), strategy,
        estimation_window=ESTIMATION_WINDOW, holding_period=HOLDING_PERIOD,
        transaction_cost_rate=TRANSACTION_COST_RATE,
    )
    pd.testing.assert_series_equal(result.backtest.net_returns, expected.net_returns)
    pd.testing.assert_frame_equal(result.backtest.target_weights, expected.target_weights)
    assert not result.backtest.net_returns.equals(result.backtest.gross_returns)
    assert result.strategy == strategy
    assert result.performance.index.to_list() == [STRATEGY_LABELS[strategy], "TOPIX"]
    pd.testing.assert_frame_equal(asset_prices, originals[0])
    pd.testing.assert_frame_equal(benchmark_prices, originals[1])


@pytest.mark.parametrize("strategy", list(STRATEGY_LABELS))
def test_dashboard_charts_and_metrics_share_benchmark_period(sample_market_prices, strategy) -> None:
    asset_prices, benchmark_prices = sample_market_prices
    benchmark_prices = benchmark_prices.iloc[520:621]
    result = run_sample_analysis(asset_prices, benchmark_prices, strategy)
    dates = benchmark_prices.index[1:]
    assert result.cumulative_returns.index.equals(dates)
    assert result.drawdowns.index.equals(dates)
    assert result.performance.attrs["start_date"] == dates[0]
    assert result.performance.attrs["end_date"] == dates[-1]
    assert result.performance.attrs["observations"] == len(dates)
    returns = {
        STRATEGY_LABELS[strategy]: result.backtest.net_returns.loc[dates],
        "TOPIX": calculate_asset_returns(benchmark_prices).iloc[:, 0],
    }
    for name, daily_returns in returns.items():
        for metric, value in performance_summary(daily_returns).items():
            assert result.performance.loc[name, metric] == pytest.approx(value)
        assert result.cumulative_returns[name].iloc[-1] == pytest.approx(
            (1.0 + daily_returns).prod() - 1.0
        )
        assert result.drawdowns[name].min() == pytest.approx(
            result.performance.loc[name, "max_drawdown"]
        )


def test_dashboard_rejects_insufficient_history(sample_market_prices) -> None:
    asset_prices, benchmark_prices = sample_market_prices
    with pytest.raises(ValueError, match="at least 567 observations"):
        run_sample_analysis(asset_prices.iloc[:550], benchmark_prices)


def test_dashboard_rejects_missing_benchmark(sample_market_prices) -> None:
    asset_prices, benchmark_prices = sample_market_prices
    with pytest.raises(ValueError, match="benchmark ticker is missing"):
        run_sample_analysis(asset_prices, benchmark_prices.rename(columns=lambda _: "bad"))


def test_dashboard_rejects_disjoint_benchmark_history(sample_market_prices) -> None:
    asset_prices, benchmark_prices = sample_market_prices
    benchmark_prices.index = pd.bdate_range("2030-01-01", periods=len(benchmark_prices))
    with pytest.raises(ValueError, match="at least two return dates"):
        run_sample_analysis(asset_prices, benchmark_prices)


@pytest.mark.parametrize("strategy", ["black_litterman", "unknown", "Equal Weight"])
def test_dashboard_rejects_unsupported_strategies(sample_market_prices, strategy) -> None:
    with pytest.raises(ValueError, match="unsupported dashboard strategy"):
        run_sample_analysis(*sample_market_prices, strategy)


def test_dashboard_default_still_uses_equal_weight(sample_market_prices) -> None:
    default = run_sample_analysis(*sample_market_prices)
    explicit = run_sample_analysis(*sample_market_prices, "equal_weight")
    assert default.strategy == "equal_weight"
    pd.testing.assert_frame_equal(default.performance, explicit.performance)
    pd.testing.assert_series_equal(default.backtest.net_returns, explicit.backtest.net_returns)
