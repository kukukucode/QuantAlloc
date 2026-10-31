"""Check stress testing uses shared OOS net returns."""

import numpy as np
import pandas as pd
import pytest

from src.benchmark import calculate_benchmark_returns
from src.dashboard import run_strategy_comparison
from src.sample_config import BENCHMARK_TICKER, STRATEGY_LABELS


def test_stress_uses_benchmark_defined_common_window(
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

    benchmark_rolling = (1.0 + aligned["TOPIX"]).rolling(21).apply(
        np.prod, raw=True
    ) - 1.0

    expected_end = benchmark_rolling.idxmin()
    end_position = aligned.index.get_loc(expected_end)
    expected_start = aligned.index[end_position - 20]

    assert result.stress.start == expected_start
    assert result.stress.end == expected_end
    assert result.stress.summary.index.tolist() == (
        list(STRATEGY_LABELS.values()) + ["TOPIX"]
    )

    event = aligned.loc[expected_start:expected_end]

    for name in aligned.columns:
        expected_return = (1.0 + event[name]).prod() - 1.0
        assert result.stress.summary.loc[name, "window_return"] == pytest.approx(
            expected_return
        )

    pd.testing.assert_frame_equal(
        result.stress.cumulative_returns,
        (1.0 + event).cumprod() - 1.0,
    )
