"""Retrospective benchmark-defined market stress analysis."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.metrics import max_drawdown


@dataclass(frozen=True)
class StressResult:
    """Stress window, summary and paths on common OOS returns."""

    start: pd.Timestamp
    end: pd.Timestamp
    summary: pd.DataFrame
    cumulative_returns: pd.DataFrame


def worst_benchmark_stress(
    returns: pd.DataFrame,
    benchmark: str = "TOPIX",
    window: int = 21,
) -> StressResult:
    """Select the worst compounded benchmark window, then compare all assets."""
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a DataFrame")
    if returns.empty:
        raise ValueError("returns must not be empty")
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("returns must use a DatetimeIndex")
    if (
        returns.index.has_duplicates
        or returns.index.hasnans
        or not returns.index.is_monotonic_increasing
    ):
        raise ValueError("returns dates must be sorted and unique")
    if returns.columns.has_duplicates:
        raise ValueError("returns columns must be unique")
    if benchmark not in returns.columns:
        raise ValueError("benchmark column is missing")
    if isinstance(window, bool) or not isinstance(window, int) or window < 2:
        raise ValueError("window must be an integer >= 2")
    if len(returns) < window:
        raise ValueError("not enough observations for stress window")

    try:
        validated = returns.apply(pd.to_numeric, errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("returns must be numeric") from exc

    if not np.isfinite(validated.to_numpy()).all():
        raise ValueError("returns must be finite")
    if (validated < -1.0).any().any():
        raise ValueError("returns must not be below -100%")

    # Retrospective selection based on TOPIX alone.
    compounded = (
        (1.0 + validated[benchmark])
        .rolling(window, min_periods=window)
        .apply(np.prod, raw=True)
    )

    end_date = compounded.idxmin()
    end_pos = validated.index.get_loc(end_date)
    start_pos = end_pos - window + 1

    event = validated.iloc[start_pos : end_pos + 1]
    paths = (1.0 + event).cumprod() - 1.0

    rows = {}

    for name in validated.columns:
        event_returns = event[name]
        ending_wealth = float((1.0 + event_returns).prod())
        recovery_days = None

        # Recovery means regaining the value immediately
        # before the stress window, measured after its end.
        if ending_wealth >= 1.0 - 1e-12:
            recovery_days = 0
        else:
            wealth = ending_wealth
            later_returns = validated[name].iloc[end_pos + 1 :]

            for days, daily_return in enumerate(later_returns, start=1):
                wealth *= 1.0 + float(daily_return)
                if wealth >= 1.0 - 1e-12:
                    recovery_days = days
                    break

        rows[name] = {
            "window_return": ending_wealth - 1.0,
            "max_drawdown": max_drawdown(event_returns),
            "recovery_days": recovery_days,
        }

    summary = pd.DataFrame.from_dict(rows, orient="index")
    summary.index.name = "Strategy"

    return StressResult(
        start=event.index[0],
        end=event.index[-1],
        summary=summary,
        cumulative_returns=paths,
    )
