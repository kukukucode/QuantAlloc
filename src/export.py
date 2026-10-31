"""Export analysis results and source prices as an auditable ZIP."""

import json
from dataclasses import asdict
from hashlib import sha256
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

from src.dashboard import SampleAnalysis, StrategyComparison
from src.dashboard_settings import DashboardSettings
from src.sample_config import BENCHMARK_TICKER, END_DATE, START_DATE
from src.sensitivity import SensitivityResult


def _evaluation_window(table: pd.DataFrame) -> dict:
    """Record the exact period used for a comparison."""
    return {
        "start_date": table.attrs["start_date"].date().isoformat(),
        "end_date": table.attrs["end_date"].date().isoformat(),
        "observations": int(table.attrs["observations"]),
    }


def create_export_archive(
    result: SampleAnalysis | StrategyComparison | SensitivityResult,
    settings: DashboardSettings,
    asset_prices: pd.DataFrame,
    benchmark_prices: pd.DataFrame,
    *,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    benchmark_ticker: str = BENCHMARK_TICKER,
) -> bytes:
    """Export an already-computed result without rerunning analysis."""
    if not isinstance(settings, DashboardSettings):
        raise TypeError("settings must be DashboardSettings")

    if not isinstance(asset_prices, pd.DataFrame):
        raise TypeError("asset_prices must be a DataFrame")

    if not isinstance(benchmark_prices, pd.DataFrame):
        raise TypeError("benchmark_prices must be a DataFrame")

    files: dict[str, bytes] = {}

    def add_csv(name: str, values: pd.DataFrame | pd.Series) -> None:
        if isinstance(values, pd.Series):
            values = values.to_frame()
        files[name] = values.to_csv(
            index=True,
            float_format="%.17g",
        ).encode("utf-8")

    add_csv("input/asset_prices.csv", asset_prices)
    add_csv("input/benchmark_prices.csv", benchmark_prices)

    manifest = {
        "schema_version": 1,
        "project": "QuantAlloc",
        "release": "v0.22",
        "settings": asdict(settings),
        "input_data": {
            "provider": "Yahoo Finance via yfinance",
            "requested_start_date": start_date,
            "requested_end_date_exclusive": end_date,
            "asset_tickers": list(asset_prices.columns),
            "benchmark_ticker": benchmark_ticker,
            "asset_observations": len(asset_prices),
            "benchmark_observations": len(benchmark_prices),
        },
        "conventions": {
            "strategy_returns": "net of modeled transaction costs",
            "benchmark_returns": "without modeled transaction costs",
            "risk_free_rate_annual": 0.0,
            "performance_return_units": "decimal fraction",
        },
    }

    if isinstance(result, SampleAnalysis):
        manifest["analysis_type"] = "single_strategy"
        manifest["strategy"] = result.strategy
        manifest["evaluation_window"] = _evaluation_window(result.performance)

        add_csv("results/performance.csv", result.performance)
        add_csv("results/tail_risk.csv", result.tail_risk)
        add_csv(
            "results/cumulative_returns.csv",
            result.cumulative_returns,
        )
        add_csv("results/drawdowns.csv", result.drawdowns)
        add_csv(
            "results/net_returns.csv",
            result.backtest.net_returns,
        )
        add_csv(
            "results/target_weights.csv",
            result.backtest.target_weights,
        )
        add_csv(
            "results/turnover.csv",
            result.backtest.turnover,
        )
        add_csv(
            "results/transaction_costs.csv",
            result.backtest.transaction_costs,
        )
        add_csv(
            "results/rebalance_periods.csv",
            result.backtest.periods,
        )

    elif isinstance(result, StrategyComparison):
        manifest["analysis_type"] = "strategy_comparison"
        manifest["strategies"] = list(result.backtests)
        manifest["evaluation_window"] = _evaluation_window(result.performance)

        add_csv("results/performance.csv", result.performance)
        add_csv("results/tail_risk.csv", result.tail_risk)
        add_csv(
            "results/cumulative_returns.csv",
            result.cumulative_returns,
        )
        add_csv("results/drawdowns.csv", result.drawdowns)
        add_csv(
            "results/factor_exposure.csv",
            result.factor_exposure,
        )
        add_csv(
            "results/stress_summary.csv",
            result.stress.summary,
        )
        add_csv(
            "results/stress_cumulative_returns.csv",
            result.stress.cumulative_returns,
        )

        for window, values in result.rolling_volatility.items():
            add_csv(
                f"results/rolling_volatility_{window}.csv",
                values,
            )

        for window, values in result.rolling_sharpe.items():
            add_csv(
                f"results/rolling_sharpe_{window}.csv",
                values,
            )

        for strategy, backtest in result.backtests.items():
            prefix = f"backtests/{strategy}"

            add_csv(
                f"{prefix}/net_returns.csv",
                backtest.net_returns,
            )
            add_csv(
                f"{prefix}/target_weights.csv",
                backtest.target_weights,
            )
            add_csv(
                f"{prefix}/turnover.csv",
                backtest.turnover,
            )
            add_csv(
                f"{prefix}/transaction_costs.csv",
                backtest.transaction_costs,
            )

    elif isinstance(result, SensitivityResult):
        manifest["analysis_type"] = "parameter_sensitivity"
        manifest["strategy"] = result.strategy
        manifest["evaluation_windows"] = {
            name: _evaluation_window(table) for name, table in result.tables.items()
        }

        for parameter, table in result.tables.items():
            add_csv(f"results/{parameter}.csv", table)

    else:
        raise TypeError("unsupported analysis result type")

    manifest["file_sha256"] = {
        name: sha256(data).hexdigest() for name, data in sorted(files.items())
    }

    output = BytesIO()

    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            archive.writestr(name, data)

        archive.writestr(
            "manifest.json",
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ).encode("utf-8"),
        )

    return output.getvalue()
