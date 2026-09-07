"""Portfolio validation and return calculations."""

import numpy as np
import pandas as pd


def validate_weights(
    weights: dict[str, float],
    tickers: list[str],
) -> pd.Series:
    """Validate long-only, fully invested portfolio weights."""
    if not weights:
        raise ValueError("weights must not be empty")

    missing_tickers = [ticker for ticker in weights if ticker not in tickers]
    if missing_tickers:
        missing = ", ".join(missing_tickers)
        raise ValueError(f"weights contain tickers without price data: {missing}")

    try:
        validated = pd.Series(weights, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("weights must be numeric") from exc

    if not np.isfinite(validated.to_numpy()).all():
        raise ValueError("weights must be finite and must not contain NaN")
    if (validated < 0).any():
        raise ValueError("weights must be non-negative")
    if not np.isclose(validated.sum(), 1.0):
        raise ValueError("weights must sum to 1.0")

    validated.name = "weight"
    return validated


def calculate_asset_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate simple daily returns for each asset."""
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError("prices must be a non-empty DataFrame")
    if len(prices) < 2:
        raise ValueError("prices must contain at least two observations")

    try:
        numeric_prices = prices.apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("prices must be numeric") from exc

    if numeric_prices.isna().any().any():
        raise ValueError("prices must not contain missing values")
    if not np.isfinite(numeric_prices.to_numpy(dtype=float)).all():
        raise ValueError("prices must contain only finite values")
    if (numeric_prices <= 0).any().any():
        raise ValueError("prices must be greater than zero")

    asset_returns = (
        numeric_prices.sort_index()
        .pct_change(fill_method=None)
        .dropna(how="any")
    )
    if asset_returns.empty:
        raise ValueError("asset returns could not be calculated")

    return asset_returns


def calculate_portfolio_returns(
    asset_returns: pd.DataFrame,
    weights: pd.Series,
) -> pd.Series:
    """Calculate weighted daily portfolio returns."""
    if not isinstance(asset_returns, pd.DataFrame) or asset_returns.empty:
        raise ValueError("asset_returns must be a non-empty DataFrame")
    if not isinstance(weights, pd.Series):
        raise TypeError("weights must be a pandas Series")

    validated_weights = validate_weights(
        weights.to_dict(),
        list(asset_returns.columns),
    )

    selected_returns = asset_returns.loc[:, validated_weights.index]
    if selected_returns.isna().any().any():
        raise ValueError("asset_returns must not contain missing values")
    if not np.isfinite(selected_returns.to_numpy(dtype=float)).all():
        raise ValueError("asset_returns must contain only finite values")

    portfolio_returns = selected_returns.dot(validated_weights)
    portfolio_returns.name = "portfolio_return"
    return portfolio_returns
