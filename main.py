"""Run the QuantAlloc v0.1 example portfolio."""

from src.data_provider import fetch_prices
from src.metrics import (
    annualized_volatility,
    cagr,
    max_drawdown,
    sharpe_ratio,
)
from src.portfolio import (
    calculate_asset_returns,
    calculate_portfolio_returns,
    validate_weights,
)

TICKERS = ["7203.T", "6758.T", "1306.T"]
WEIGHTS = {
    "7203.T": 0.40,
    "6758.T": 0.30,
    "1306.T": 0.30,
}
START_DATE = "2021-01-01"
END_DATE = "2026-01-01"


def main() -> None:
    """Fetch prices, calculate returns, and print portfolio metrics."""
    prices = fetch_prices(TICKERS, START_DATE, END_DATE)
    asset_returns = calculate_asset_returns(prices)
    weights = validate_weights(WEIGHTS, list(prices.columns))
    portfolio_returns = calculate_portfolio_returns(asset_returns, weights)

    print(f"CAGR          : {cagr(portfolio_returns):.2%}")
    print(
        "Volatility    : "
        f"{annualized_volatility(portfolio_returns):.2%}"
    )
    print(f"Sharpe Ratio  : {sharpe_ratio(portfolio_returns):.2f}")
    print(f"Max Drawdown  : {max_drawdown(portfolio_returns):.2%}")


if __name__ == "__main__":
    main()
