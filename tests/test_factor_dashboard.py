"""Check market exposure uses common OOS net returns."""

import pandas as pd
import pytest

from src.benchmark import calculate_benchmark_returns
from src.dashboard import run_strategy_comparison
from src.sample_config import (
    BENCHMARK_TICKER,
    STRATEGY_LABELS,
)


def test_factor_exposure_matches_common_oos_regression(
    sample_market_prices,
) -> None:
    asset_prices, benchmark_prices = sample_market_prices
    benchmark_prices = benchmark_prices.iloc[520:621]

    result = run_strategy_comparison(asset_prices, benchmark_prices)

    dates = result.cumulative_returns.index

    aligned = pd.DataFrame(
        {
            STRATEGY_LABELS[strategy]: backtest.net_returns.loc[dates]
            for strategy, backtest in result.backtests.items()
        }
    )

    aligned["TOPIX"] = calculate_benchmark_returns(
        benchmark_prices, BENCHMARK_TICKER
    ).loc[dates]

    exposure = result.factor_exposure

    assert exposure.index.tolist() == (list(STRATEGY_LABELS.values()) + ["TOPIX"])
    assert exposure.columns.tolist() == ["beta", "alpha_annual", "r_squared"]

    market = aligned["TOPIX"]

    for name in aligned.columns:
        strategy = aligned[name]
        expected_beta = strategy.cov(market) / market.var()
        expected_alpha = (strategy.mean() - expected_beta * market.mean()) * 252
        expected_r_squared = strategy.corr(market) ** 2

        assert exposure.loc[name, "beta"] == pytest.approx(expected_beta)

        assert exposure.loc[name, "alpha_annual"] == pytest.approx(expected_alpha)

        assert exposure.loc[name, "r_squared"] == pytest.approx(expected_r_squared)
