"""Run the QuantAlloc portfolio and benchmark comparison."""

import pandas as pd

from src.backtest import walk_forward_backtest
from src.benchmark import (
    calculate_benchmark_returns,
    compare_with_benchmark,
)
from src.data_provider import fetch_prices
from src.diversification import (
    correlation_matrix,
    covariance_matrix,
    effective_number_of_assets,
    hhi,
)
from src.evaluation import compare_strategies
from src.optimization import (
    efficient_frontier,
    expected_returns,
    maximum_sharpe_weights,
    minimum_variance_weights,
    portfolio_performance,
)
from src.portfolio import (
    calculate_asset_returns,
    calculate_portfolio_returns,
    validate_weights,
)
from src.risk import risk_contribution

WEIGHTS = {
    "7203.T": 0.10,
    "6758.T": 0.10,
    "8306.T": 0.10,
    "9432.T": 0.10,
    "8058.T": 0.10,
    "7974.T": 0.10,
    "3003.T": 0.10,
    "6501.T": 0.10,
    "9983.T": 0.10,
    "4661.T": 0.10,
}
TICKERS = list(WEIGHTS)
BENCHMARK_TICKER = "1306.T"
START_DATE = "2021-01-01"
END_DATE = "2026-01-01"


def print_comparison(
    result: dict[str, dict[str, float] | object],
) -> None:
    """Print portfolio and TOPIX metrics side by side."""
    portfolio = result["portfolio"]
    benchmark = result["benchmark"]
    if not isinstance(portfolio, dict) or not isinstance(benchmark, dict):
        raise TypeError("comparison result does not contain metric summaries")

    rows = (
        ("CAGR", "cagr", True),
        ("Volatility", "volatility", True),
        ("Sharpe Ratio", "sharpe", False),
        ("Max Drawdown", "max_drawdown", True),
    )

    print(f"{'':<18}{'Portfolio':>14}{'TOPIX':>14}")
    for label, key, is_percentage in rows:
        portfolio_value = portfolio[key]
        benchmark_value = benchmark[key]
        if is_percentage:
            portfolio_text = f"{portfolio_value:.2%}"
            benchmark_text = f"{benchmark_value:.2%}"
        else:
            portfolio_text = f"{portfolio_value:.2f}"
            benchmark_text = f"{benchmark_value:.2f}"
        print(f"{label:<18}{portfolio_text:>14}{benchmark_text:>14}")


def print_diversification_analysis(
    asset_returns: pd.DataFrame,
    weights: pd.Series,
    covariance: pd.DataFrame,
) -> None:
    """Print correlation, covariance, and concentration measures."""
    print("\nCorrelation Matrix")
    print(correlation_matrix(asset_returns).round(2).to_string())

    print("\nAnnualized Covariance Matrix")
    print(covariance.round(4).to_string())

    print("\nConcentration")
    print(f"HHI                        : {hhi(weights):.4f}")
    print(
        "Effective Number of Assets : "
        f"{effective_number_of_assets(weights):.2f}"
    )

    contributions = risk_contribution(weights, covariance)
    risk_table = contributions[["weight", "risk_contribution_pct"]].copy()
    risk_table.columns = ["Weight", "Risk Contribution"]
    print("\nRisk Contribution")
    print(risk_table.map(lambda value: f"{value:.2%}").to_string())


def print_optimization_analysis(
    current_weights: pd.Series,
    minimum_weights: pd.Series,
    maximum_sharpe: pd.Series,
    historical_returns: pd.Series,
    covariance: pd.DataFrame,
    frontier: pd.DataFrame,
) -> None:
    """Print optimized allocations, performance, and efficient frontier."""
    allocations = pd.DataFrame(
        {
            "Current": current_weights,
            "Minimum Variance": minimum_weights,
            "Maximum Sharpe": maximum_sharpe,
        }
    )
    print("\nPortfolio Weights")
    print(allocations.map(lambda value: f"{value:.2%}").to_string())

    portfolios = {
        "Current": current_weights,
        "Minimum Variance": minimum_weights,
        "Maximum Sharpe": maximum_sharpe,
    }
    performance = {
        name: portfolio_performance(
            portfolio_weights,
            historical_returns,
            covariance,
        )
        for name, portfolio_weights in portfolios.items()
    }

    print("\nExpected Performance")
    print(f"{'':<18}{'Current':>14}{'Min Variance':>16}{'Max Sharpe':>14}")
    rows = (
        ("Return", "return", True),
        ("Volatility", "volatility", True),
        ("Sharpe Ratio", "sharpe", False),
    )
    for label, key, is_percentage in rows:
        values = [performance[name][key] for name in portfolios]
        formatted = [
            f"{value:.2%}" if is_percentage else f"{value:.2f}"
            for value in values
        ]
        print(
            f"{label:<18}{formatted[0]:>14}"
            f"{formatted[1]:>16}{formatted[2]:>14}"
        )

    frontier_table = frontier.copy()
    frontier_table.columns = ["Return", "Volatility"]
    print("\nEfficient Frontier")
    print(frontier_table.map(lambda value: f"{value:.2%}").to_string(index=False))


def print_strategy_comparison(
    comparison: pd.DataFrame,
) -> None:
    """Print an aligned out-of-sample strategy comparison."""
    print("\nOut-of-Sample Strategy Comparison")
    print(
        f"{'Strategy':<20}{'CAGR':>10}{'Vol':>10}"
        f"{'Sharpe':>10}{'MDD':>10}"
    )
    for name, values in comparison.iterrows():
        print(
            f"{name:<20}{values['cagr']:>9.2%}"
            f"{values['volatility']:>9.2%}"
            f"{values['sharpe']:>10.2f}"
            f"{values['max_drawdown']:>9.2%}"
        )

    start_date = comparison.attrs["start_date"].date()
    end_date = comparison.attrs["end_date"].date()
    print(f"\nOOS Period: {start_date} -> {end_date}")
    print(f"Observations: {comparison.attrs['observations']}")


def main() -> None:
    """Compare the example portfolio with the TOPIX benchmark."""
    prices = fetch_prices(TICKERS, START_DATE, END_DATE)
    asset_returns = calculate_asset_returns(prices)
    weights = validate_weights(WEIGHTS, list(prices.columns))
    portfolio_returns = calculate_portfolio_returns(asset_returns, weights)

    benchmark_prices = fetch_prices(
        [BENCHMARK_TICKER],
        START_DATE,
        END_DATE,
    )
    benchmark_returns = calculate_benchmark_returns(
        benchmark_prices,
        BENCHMARK_TICKER,
    )

    result = compare_with_benchmark(
        portfolio_returns,
        benchmark_returns,
    )
    annualized_covariance = covariance_matrix(asset_returns)
    historical_returns = expected_returns(asset_returns)
    minimum_weights = minimum_variance_weights(annualized_covariance)
    maximum_sharpe = maximum_sharpe_weights(
        historical_returns,
        annualized_covariance,
    )
    frontier = efficient_frontier(
        historical_returns,
        annualized_covariance,
        points=10,
    )
    backtest_results = {
        strategy: walk_forward_backtest(asset_returns, strategy)
        for strategy in (
            "equal_weight",
            "minimum_variance",
            "maximum_sharpe",
        )
    }
    comparison = compare_strategies(
        {
            "Equal Weight": backtest_results["equal_weight"].returns,
            "Minimum Variance": backtest_results[
                "minimum_variance"
            ].returns,
            "Maximum Sharpe": backtest_results["maximum_sharpe"].returns,
            "TOPIX": benchmark_returns,
        }
    )

    print_comparison(result)
    print_diversification_analysis(
        asset_returns,
        weights,
        annualized_covariance,
    )
    print_optimization_analysis(
        weights,
        minimum_weights,
        maximum_sharpe,
        historical_returns,
        annualized_covariance,
        frontier,
    )
    print_strategy_comparison(comparison)


if __name__ == "__main__":
    main()
