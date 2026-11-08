"""Validate simultaneous strategy comparisons."""

import pandas as pd
import pytest

import src.backtest as backtest_engine
from src.backtest import walk_forward_backtest
from src.benchmark import calculate_benchmark_returns
from src.dashboard import run_strategy_comparison
from src.evaluation import performance_summary
from src.portfolio import calculate_asset_returns
from src.risk import conditional_value_at_risk, value_at_risk
from src.sample_config import BENCHMARK_TICKER, STRATEGY_LABELS


def test_all_strategies_share_common_oos_dates(
    sample_market_prices,
) -> None:
    asset_prices, benchmark_prices = sample_market_prices
    benchmark_prices = benchmark_prices.iloc[520:621]

    result = run_strategy_comparison(asset_prices, benchmark_prices)

    dates = benchmark_prices.index[1:]
    labels = list(STRATEGY_LABELS.values()) + ["TOPIX"]

    assert list(result.backtests) == list(STRATEGY_LABELS)
    assert result.performance.index.tolist() == labels
    assert result.tail_risk.index.tolist() == labels

    assert result.cumulative_returns.index.equals(dates)
    assert result.drawdowns.index.equals(dates)

    assert result.performance.attrs["start_date"] == dates[0]
    assert result.performance.attrs["end_date"] == dates[-1]
    assert result.performance.attrs["observations"] == len(dates)

    returns = {
        STRATEGY_LABELS[name]: backtest.net_returns.loc[dates]
        for name, backtest in result.backtests.items()
    }
    returns["TOPIX"] = calculate_benchmark_returns(
        benchmark_prices, BENCHMARK_TICKER
    ).loc[dates]

    for label, daily_returns in returns.items():
        metrics = performance_summary(daily_returns)

        for metric, expected in metrics.items():
            assert result.performance.loc[label, metric] == pytest.approx(
                expected, nan_ok=True
            )

        assert result.tail_risk.loc[label, "var_95"] == pytest.approx(
            value_at_risk(daily_returns)
        )

        assert result.tail_risk.loc[label, "cvar_95"] == pytest.approx(
            conditional_value_at_risk(daily_returns)
        )

        expected_wealth = (1.0 + daily_returns).cumprod()
        expected_cumulative = expected_wealth - 1.0
        expected_drawdown = (
            expected_wealth / expected_wealth.cummax().clip(lower=1.0) - 1.0
        )

        pd.testing.assert_series_equal(
            result.cumulative_returns[label],
            expected_cumulative.rename(label),
        )
        pd.testing.assert_series_equal(
            result.drawdowns[label],
            expected_drawdown.rename(label),
        )


def test_comparison_rejects_missing_benchmark(
    sample_market_prices,
) -> None:
    asset_prices, benchmark_prices = sample_market_prices
    benchmark_prices = benchmark_prices.rename(columns=lambda _: "bad")

    with pytest.raises(ValueError, match="benchmark ticker is missing"):
        run_strategy_comparison(asset_prices, benchmark_prices)


def test_comparison_reuses_one_covariance_per_training_window(
    sample_market_prices, monkeypatch
) -> None:
    asset_prices, benchmark_prices = sample_market_prices
    asset_returns = calculate_asset_returns(asset_prices)
    calls = []
    original_estimate = backtest_engine.estimate_covariance

    def track_estimate(*args, **kwargs):
        calls.append((args[0].index[0], args[0].index[-1], kwargs["method"]))
        return original_estimate(*args, **kwargs)

    monkeypatch.setattr(backtest_engine, "estimate_covariance", track_estimate)
    comparison = run_strategy_comparison(asset_prices, benchmark_prices)

    expected_windows = len(range(504, len(asset_returns) - 63 + 1, 63))
    assert len(calls) == expected_windows
    assert all(method == "sample" for _, _, method in calls)

    for strategy, optimized in comparison.backtests.items():
        baseline = walk_forward_backtest(
            asset_returns,
            strategy,
            estimation_window=504,
            holding_period=63,
            transaction_cost_rate=0.001,
        )
        pd.testing.assert_series_equal(optimized.gross_returns, baseline.gross_returns)
        pd.testing.assert_series_equal(optimized.net_returns, baseline.net_returns)
        pd.testing.assert_frame_equal(optimized.target_weights, baseline.target_weights)
        pd.testing.assert_frame_equal(
            optimized.pre_rebalance_weights, baseline.pre_rebalance_weights
        )
        pd.testing.assert_series_equal(optimized.turnover, baseline.turnover)
        pd.testing.assert_series_equal(
            optimized.transaction_costs, baseline.transaction_costs
        )
        pd.testing.assert_frame_equal(optimized.periods, baseline.periods)
