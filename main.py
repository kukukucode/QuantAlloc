"""Run the QuantAlloc portfolio and benchmark comparison."""

import pandas as pd

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
from src.portfolio import (
    calculate_asset_returns,
    calculate_portfolio_returns,
    validate_weights,
)

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
) -> None:
    """Print correlation, covariance, and concentration measures."""
    print("\nCorrelation Matrix")
    print(correlation_matrix(asset_returns).round(2).to_string())

    print("\nAnnualized Covariance Matrix")
    print(covariance_matrix(asset_returns).round(4).to_string())

    print("\nConcentration")
    print(f"HHI                        : {hhi(weights):.4f}")
    print(
        "Effective Number of Assets : "
        f"{effective_number_of_assets(weights):.2f}"
    )


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
    print_comparison(result)
    print_diversification_analysis(asset_returns, weights)


if __name__ == "__main__":
    main()
