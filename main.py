"""Run the QuantAlloc portfolio and benchmark comparison."""

from src.benchmark import (
    calculate_benchmark_returns,
    compare_with_benchmark,
)
from src.data_provider import fetch_prices
from src.portfolio import (
    calculate_asset_returns,
    calculate_portfolio_returns,
    validate_weights,
)

TICKERS = ["7203.T", "6758.T", "8306.T"]
WEIGHTS = {
    "7203.T": 0.40,
    "6758.T": 0.30,
    "8306.T": 0.30,
}
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


if __name__ == "__main__":
    main()
