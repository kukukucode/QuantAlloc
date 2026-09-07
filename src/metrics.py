"""Portfolio performance metrics."""

import math

import numpy as np
import pandas as pd


def _validated_returns(returns: pd.Series) -> pd.Series:
    if not isinstance(returns, pd.Series) or returns.empty:
        raise ValueError("returns must be a non-empty pandas Series")

    try:
        validated = pd.to_numeric(returns, errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("returns must be numeric") from exc

    if not np.isfinite(validated.to_numpy()).all():
        raise ValueError("returns must be finite and must not contain NaN")
    if (validated < -1.0).any():
        raise ValueError("returns must not be less than -100%")

    return validated


def _validate_periods_per_year(periods_per_year: int) -> None:
    if isinstance(periods_per_year, bool) or periods_per_year <= 0:
        raise ValueError("periods_per_year must be a positive integer")


def wealth_index(
    returns: pd.Series,
    initial_value: float = 1.0,
) -> pd.Series:
    """Calculate cumulative portfolio value from periodic returns."""
    validated = _validated_returns(returns)
    if not math.isfinite(initial_value) or initial_value <= 0:
        raise ValueError("initial_value must be a positive finite number")

    wealth = initial_value * (1.0 + validated).cumprod()
    wealth.name = "wealth"
    return wealth


def cagr(returns: pd.Series) -> float:
    """Calculate compound annual growth using the actual date span."""
    validated = _validated_returns(returns)
    if not isinstance(validated.index, pd.DatetimeIndex):
        raise TypeError("returns must use a DatetimeIndex")
    if len(validated) < 2:
        raise ValueError("returns must contain at least two dated observations")
    if not validated.index.is_monotonic_increasing:
        raise ValueError("returns index must be sorted in ascending order")

    elapsed_days = (validated.index[-1] - validated.index[0]).days
    if elapsed_days <= 0:
        raise ValueError("returns must span more than one calendar day")

    years = elapsed_days / 365.25
    ending_value = float(wealth_index(validated).iloc[-1])
    if ending_value == 0.0:
        return -1.0

    return ending_value ** (1.0 / years) - 1.0


def annualized_volatility(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Calculate annualized sample volatility."""
    validated = _validated_returns(returns)
    _validate_periods_per_year(periods_per_year)
    if len(validated) < 2:
        raise ValueError("returns must contain at least two observations")

    return float(validated.std(ddof=1) * math.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Calculate annualized Sharpe ratio using an annual risk-free rate."""
    validated = _validated_returns(returns)
    _validate_periods_per_year(periods_per_year)
    if not math.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be finite")

    volatility = annualized_volatility(validated, periods_per_year)
    if np.isclose(volatility, 0.0):
        return float("nan")

    annualized_return = float(validated.mean() * periods_per_year)
    return (annualized_return - risk_free_rate) / volatility


def max_drawdown(returns: pd.Series) -> float:
    """Calculate the largest peak-to-trough decline."""
    wealth = wealth_index(returns)
    running_peak = wealth.cummax().clip(lower=1.0)
    drawdown = wealth / running_peak - 1.0
    return float(drawdown.min())
