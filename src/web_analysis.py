"""Coverage-aware evaluation of a portfolio over user-selected dates."""

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from src.backtest import walk_forward_backtest
from src.dashboard_settings import DashboardSettings


class NoEvaluationData(ValueError):
    """No portfolio and benchmark return can be measured in the request."""


@dataclass(frozen=True)
class WebAnalysis:
    returns: pd.DataFrame
    strategy: str
    entry_dates: dict[str, pd.Timestamp]
    unavailable: list[str]
    skipped_sessions: int


def run_web_analysis(
    asset_prices: pd.DataFrame,
    benchmark_prices: pd.Series,
    strategy: str,
    start_date: date,
    end_date: date,
    settings: DashboardSettings,
) -> WebAnalysis:
    """Evaluate complete price intervals, admitting newly quoted assets in Equal Weight."""
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    benchmark = benchmark_prices.where(
        np.isfinite(benchmark_prices) & (benchmark_prices > 0)
    )
    benchmark = benchmark.loc[:end].dropna()
    assets = asset_prices.reindex(benchmark.index)
    assets = assets.where(np.isfinite(assets) & (assets > 0))

    # For each security, pre-listing closes may be absent. Once a close is
    # observed, subsequent valuation dates require a close for that security.
    priced_in_window = assets.loc[start:].notna().any()
    assets = assets.loc[:, priced_in_window]
    observed = assets.notna()
    valuation_dates = assets.index[(observed | ~observed.cummax()).all(axis=1)]
    asset_returns = assets.loc[valuation_dates].pct_change(fill_method=None)
    benchmark_returns = benchmark.loc[valuation_dates].pct_change(fill_method=None)
    start_position = benchmark.index.searchsorted(start)
    if start_position >= len(benchmark):
        raise NoEvaluationData("no benchmark closes in requested dates")
    previous_session = benchmark.index[max(0, start_position - 1)]
    # A return ending within the request must not include an earlier price gap.
    within_window = (valuation_dates >= start) & (
        valuation_dates.to_series().shift() >= previous_session
    )
    evaluation_returns = asset_returns.loc[within_window].dropna(how="all")
    if evaluation_returns.empty:
        raise NoEvaluationData(
            "no comparable holding-period returns in requested dates"
        )

    entry_dates: dict[str, pd.Timestamp] = {}
    training_days = settings.estimation_window
    complete_returns = asset_returns.dropna(how="any")
    first_test = max(
        training_days, complete_returns.index.searchsorted(evaluation_returns.index[0])
    )
    if (
        strategy != "equal_weight"
        and priced_in_window.all()
        and first_test < len(complete_returns)
    ):
        backtest_inputs = complete_returns.iloc[first_test - training_days :]
    else:
        strategy = "equal_weight"
        training_days = 0
        entry_dates = {
            symbol: day
            for symbol, values in evaluation_returns.items()
            if (day := values.first_valid_index()) is not None
        }
        # Zero returns in unpriced columns have zero portfolio weight until entry.
        backtest_inputs = evaluation_returns.loc[:, list(entry_dates)].fillna(0.0)

    result = walk_forward_backtest(
        backtest_inputs,
        strategy,
        estimation_window=training_days,
        holding_period=settings.holding_period,
        transaction_cost_rate=settings.transaction_cost_rate,
        include_partial_period=True,
        asset_entry_dates=entry_dates if strategy == "equal_weight" else None,
    )
    evaluated = result.net_returns.index
    returns = pd.DataFrame(
        {"strategy": result.net_returns, "benchmark": benchmark_returns.loc[evaluated]}
    )
    unavailable = [symbol for symbol in asset_prices if symbol not in backtest_inputs]
    skipped = len(benchmark.loc[evaluated[0] : evaluated[-1]]) - len(evaluated)
    return WebAnalysis(returns, strategy, entry_dates, unavailable, skipped)
