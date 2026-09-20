"""Core Black-Litterman calculations."""

import math

import pandas as pd

from src.portfolio import validate_weights
from src.risk import validate_covariance


def implied_equilibrium_returns(
    market_weights: pd.Series,
    covariance: pd.DataFrame,
    risk_aversion: float,
) -> pd.Series:
    """Calculate market equilibrium returns as risk_aversion * covariance * weights."""
    if not isinstance(market_weights, pd.Series) or market_weights.empty:
        raise ValueError("market_weights must be a non-empty Series")
    if market_weights.index.has_duplicates:
        raise ValueError("market_weights index must be unique")
    if isinstance(risk_aversion, bool):
        raise ValueError("risk_aversion must be a positive finite number")
    try:
        validated_risk_aversion = float(risk_aversion)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "risk_aversion must be a positive finite number"
        ) from exc
    if (
        not math.isfinite(validated_risk_aversion)
        or validated_risk_aversion <= 0.0
    ):
        raise ValueError("risk_aversion must be a positive finite number")

    validated_covariance = validate_covariance(covariance)
    if set(market_weights.index) != set(validated_covariance.columns):
        raise ValueError("market_weights and covariance assets must match")

    validated_weights = validate_weights(
        market_weights.to_dict(),
        list(validated_covariance.columns),
    ).reindex(validated_covariance.columns)
    implied_returns = (
        validated_covariance.dot(validated_weights)
        * validated_risk_aversion
    )
    implied_returns.name = "implied_equilibrium_return"
    return implied_returns
