"""Pure analysis orchestration for the strategy-selection dashboard."""

from dataclasses import dataclass

import pandas as pd

from src.backtest import BacktestResult, walk_forward_backtest
from src.benchmark import calculate_benchmark_returns
from src.evaluation import _align_strategy_returns, compare_strategies
from src.factor import market_factor_exposure
from src.portfolio import calculate_asset_returns
from src.risk import conditional_value_at_risk, value_at_risk
from src.rolling import rolling_risk_metrics
from src.sample_config import (
    BENCHMARK_TICKER,
    ESTIMATION_WINDOW,
    HOLDING_PERIOD,
    STRATEGY_LABELS,
    TRANSACTION_COST_RATE,
)
from src.sensitivity import SensitivityResult, run_parameter_sensitivity
from src.stress import StressResult, worst_benchmark_stress
from src.visualization import cumulative_return_paths, drawdown_paths


@dataclass(frozen=True)
class SampleAnalysis:
    """Already-computed results for display, without any Streamlit dependency."""

    backtest: BacktestResult
    performance: pd.DataFrame
    cumulative_returns: pd.DataFrame
    drawdowns: pd.DataFrame
    tail_risk: pd.DataFrame
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
    returns = {
        STRATEGY_LABELS[strategy]: backtest.net_returns,
        "TOPIX": benchmark_returns,
    }
    aligned_returns = _align_strategy_returns(returns)
    tail_risk = pd.DataFrame.from_dict(
        {
            name: {
                "var_95": value_at_risk(aligned_returns[name]),
                "cvar_95": conditional_value_at_risk(aligned_returns[name]),
            }
            for name in aligned_returns.columns
        },
        orient="index",
    )
    tail_risk.index.name = "Strategy"

    return SampleAnalysis(
        backtest=backtest,
        performance=compare_strategies(returns),
        cumulative_returns=cumulative_return_paths(returns),
        drawdowns=drawdown_paths(returns),
        tail_risk=tail_risk,
        strategy=strategy,
    )


@dataclass(frozen=True)
class StrategyComparison:
    """Results for all supported strategies on one common OOS period."""

    backtests: dict[str, BacktestResult]
    performance: pd.DataFrame
    cumulative_returns: pd.DataFrame
    drawdowns: pd.DataFrame
    tail_risk: pd.DataFrame
    rolling_volatility: dict[int, pd.DataFrame]
    rolling_sharpe: dict[int, pd.DataFrame]
    stress: StressResult
    factor_exposure: pd.DataFrame


def run_strategy_comparison(
    asset_prices: pd.DataFrame,
    benchmark_prices: pd.DataFrame,
) -> StrategyComparison:
    """Compare all dashboard strategies and TOPIX on identical dates."""
    asset_returns = calculate_asset_returns(asset_prices)
    benchmark_returns = calculate_benchmark_returns(benchmark_prices, BENCHMARK_TICKER)

    backtests = {
        strategy: walk_forward_backtest(
            asset_returns,
            strategy,
            estimation_window=ESTIMATION_WINDOW,
            holding_period=HOLDING_PERIOD,
            transaction_cost_rate=TRANSACTION_COST_RATE,
        )
        for strategy in STRATEGY_LABELS
    }

    returns = {
        STRATEGY_LABELS[strategy]: result.net_returns
        for strategy, result in backtests.items()
    }
    returns["TOPIX"] = benchmark_returns

    aligned = _align_strategy_returns(returns)

    tail_risk = pd.DataFrame.from_dict(
        {
            name: {
                "var_95": value_at_risk(aligned[name]),
                "cvar_95": conditional_value_at_risk(aligned[name]),
            }
            for name in aligned.columns
        },
        orient="index",
    )
    tail_risk.index.name = "Strategy"

    rolling_stats = {
        window: rolling_risk_metrics(aligned, window) for window in (63, 126, 252)
    }

    stress = worst_benchmark_stress(aligned, window=21)
    factor_exposure = market_factor_exposure(aligned)

    return StrategyComparison(
        backtests=backtests,
        performance=compare_strategies(returns),
        cumulative_returns=cumulative_return_paths(returns),
        drawdowns=drawdown_paths(returns),
        tail_risk=tail_risk,
        stress=stress,
        factor_exposure=factor_exposure,
        rolling_volatility={
            window: metrics[0] for window, metrics in rolling_stats.items()
        },
        rolling_sharpe={
            window: metrics[1] for window, metrics in rolling_stats.items()
        },
    )


def run_sample_sensitivity(
    asset_prices: pd.DataFrame,
    strategy: str = "equal_weight",
) -> SensitivityResult:
    """Analyze parameter sensitivity on existing asset price data."""
    asset_returns = calculate_asset_returns(asset_prices)
    return run_parameter_sensitivity(asset_returns, strategy)
