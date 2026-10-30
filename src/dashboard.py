"""Pure analysis orchestration for the strategy-selection dashboard."""

from dataclasses import dataclass

import pandas as pd

from src.backtest import BacktestResult, walk_forward_backtest
from src.benchmark import calculate_benchmark_returns
from src.evaluation import compare_strategies
from src.portfolio import calculate_asset_returns
from src.sample_config import (
    BENCHMARK_TICKER,
    ESTIMATION_WINDOW,
    HOLDING_PERIOD,
    STRATEGY_LABELS,
    TRANSACTION_COST_RATE,
)
from src.visualization import cumulative_return_paths, drawdown_paths


@dataclass(frozen=True)
class SampleAnalysis:
    """Already-computed results for display, without any Streamlit dependency."""

    backtest: BacktestResult
    performance: pd.DataFrame
    cumulative_returns: pd.DataFrame
    drawdowns: pd.DataFrame
    strategy: str = "equal_weight"


def run_sample_analysis(
    asset_prices: pd.DataFrame,
    benchmark_prices: pd.DataFrame,
    strategy: str = "equal_weight",
) -> SampleAnalysis:
    """Reuse the existing engine and metrics on one common OOS period."""
    if strategy not in STRATEGY_LABELS:
        raise ValueError(f"unsupported dashboard strategy: {strategy}")
    asset_returns = calculate_asset_returns(asset_prices)
    benchmark_returns = calculate_benchmark_returns(benchmark_prices, BENCHMARK_TICKER)
    backtest = walk_forward_backtest(
        asset_returns,
        strategy,
        estimation_window=ESTIMATION_WINDOW,
        holding_period=HOLDING_PERIOD,
        transaction_cost_rate=TRANSACTION_COST_RATE,
    )
    returns = {STRATEGY_LABELS[strategy]: backtest.net_returns, "TOPIX": benchmark_returns}
    return SampleAnalysis(
        backtest=backtest,
        performance=compare_strategies(returns),
        cumulative_returns=cumulative_return_paths(returns),
        drawdowns=drawdown_paths(returns),
        strategy=strategy,
    )
