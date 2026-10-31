"""Verify that dashboard tail risk uses common OOS returns."""

import pytest

from src.benchmark import calculate_benchmark_returns
from src.dashboard import run_sample_analysis
from src.risk import conditional_value_at_risk, value_at_risk
from src.sample_config import BENCHMARK_TICKER, STRATEGY_LABELS


@pytest.mark.parametrize("strategy", list(STRATEGY_LABELS))
def test_tail_risk_matches_common_oos_returns(sample_market_prices, strategy) -> None:
    asset_prices, benchmark_prices = sample_market_prices

    # Shorter benchmark history exposes incorrect date alignment.
    benchmark_prices = benchmark_prices.iloc[520:621]

    result = run_sample_analysis(asset_prices, benchmark_prices, strategy)

    dates = result.cumulative_returns.index
    name = STRATEGY_LABELS[strategy]
    strategy_returns = result.backtest.net_returns.loc[dates]
    benchmark_returns = calculate_benchmark_returns(
        benchmark_prices, BENCHMARK_TICKER
    ).loc[dates]

    expected = {
        name: strategy_returns,
        "TOPIX": benchmark_returns,
    }

    assert result.tail_risk.index.tolist() == [name, "TOPIX"]
    assert result.tail_risk.columns.tolist() == ["var_95", "cvar_95"]

    for label, returns in expected.items():
        assert result.tail_risk.loc[label, "var_95"] == pytest.approx(
            value_at_risk(returns)
        )

        assert result.tail_risk.loc[label, "cvar_95"] == pytest.approx(
            conditional_value_at_risk(returns)
        )
