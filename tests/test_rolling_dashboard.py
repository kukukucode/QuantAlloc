"""Validate rolling risk on common OOS returns."""

import math

import pandas as pd

from src.benchmark import calculate_benchmark_returns
from src.dashboard import run_strategy_comparison
from src.sample_config import BENCHMARK_TICKER, STRATEGY_LABELS


def test_rolling_metrics_share_oos_dates(sample_market_prices) -> None:
    asset_prices, benchmark_prices = sample_market_prices
    benchmark_prices = benchmark_prices.iloc[520:621]

    result = run_strategy_comparison(asset_prices, benchmark_prices)

    dates = result.cumulative_returns.index
    aligned = pd.DataFrame(
        {
            STRATEGY_LABELS[name]: backtest.net_returns.loc[dates]
            for name, backtest in result.backtests.items()
        }
    )
    aligned["TOPIX"] = calculate_benchmark_returns(
        benchmark_prices, BENCHMARK_TICKER
    ).loc[dates]

    assert set(result.rolling_volatility) == {63, 126, 252}
    assert set(result.rolling_sharpe) == {63, 126, 252}

    for window in (63, 126, 252):
        vol = result.rolling_volatility[window]
        sharpe = result.rolling_sharpe[window]

        rolling = aligned.rolling(window=window, min_periods=window)
        expected_vol = rolling.std(ddof=1) * math.sqrt(252)
        expected_sharpe = (
            rolling.mean() * 252 / expected_vol.where(expected_vol > 1e-12)
        )

        pd.testing.assert_frame_equal(vol, expected_vol)
        pd.testing.assert_frame_equal(sharpe, expected_sharpe)

    assert result.rolling_volatility[126].isna().all().all()
    assert result.rolling_sharpe[252].isna().all().all()
