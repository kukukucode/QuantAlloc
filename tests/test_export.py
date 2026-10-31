"""Verify ZIP contents, input snapshots and export metadata."""

import json
from hashlib import sha256
from io import BytesIO
from zipfile import ZipFile

import pandas as pd
import pytest

from src.dashboard import (
    run_sample_analysis,
    run_sample_sensitivity,
    run_strategy_comparison,
)
from src.dashboard_settings import DashboardSettings
from src.export import create_export_archive


def inspect_archive(payload):
    with ZipFile(BytesIO(payload)) as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read("manifest.json"))

        assert "input/asset_prices.csv" in names
        assert "input/benchmark_prices.csv" in names

        assert set(manifest["file_sha256"]) == (names - {"manifest.json"})

        for name, digest in manifest["file_sha256"].items():
            assert sha256(archive.read(name)).hexdigest() == digest

        files = {name: archive.read(name) for name in names}

    return manifest, files


def test_single_strategy_archive(sample_market_prices) -> None:
    asset_prices, benchmark_prices = sample_market_prices

    settings = DashboardSettings(
        estimation_window=252,
        holding_period=126,
        transaction_cost_rate=0.003,
    )

    result = run_sample_analysis(
        asset_prices,
        benchmark_prices,
        settings=settings,
    )

    payload = create_export_archive(
        result,
        settings,
        asset_prices,
        benchmark_prices,
    )

    manifest, files = inspect_archive(payload)

    assert manifest["analysis_type"] == "single_strategy"
    assert manifest["strategy"] == "equal_weight"
    assert manifest["settings"]["estimation_window"] == 252
    assert manifest["settings"]["holding_period"] == 126
    assert manifest["settings"]["transaction_cost_rate"] == 0.003
    assert (
        manifest["evaluation_window"]["observations"]
        == (result.performance.attrs["observations"])
    )

    actual_prices = pd.read_csv(
        BytesIO(files["input/asset_prices.csv"]),
        index_col=0,
    )
    assert actual_prices.shape == asset_prices.shape
    assert actual_prices.columns.tolist() == asset_prices.columns.tolist()

    exported = pd.read_csv(
        BytesIO(files["results/performance.csv"]),
        index_col=0,
    )

    assert exported.loc["Equal Weight", "cagr"] == pytest.approx(
        result.performance.loc["Equal Weight", "cagr"]
    )

    assert "results/rebalance_periods.csv" in files
    assert b"ChatGPT" not in files["manifest.json"]
    assert b"Codex" not in files["manifest.json"]


def test_comparison_archive(sample_market_prices) -> None:
    asset_prices, benchmark_prices = sample_market_prices

    result = run_strategy_comparison(asset_prices, benchmark_prices)

    manifest, files = inspect_archive(
        create_export_archive(
            result,
            DashboardSettings(),
            asset_prices,
            benchmark_prices,
        )
    )

    assert manifest["analysis_type"] == "strategy_comparison"
    assert len(manifest["strategies"]) == 4
    assert "results/factor_exposure.csv" in files
    assert "results/stress_summary.csv" in files
    assert "results/rolling_volatility_63.csv" in files
    assert "backtests/equal_weight/net_returns.csv" in files


def test_sensitivity_archive(sample_market_prices) -> None:
    asset_prices, benchmark_prices = sample_market_prices

    result = run_sample_sensitivity(asset_prices, "equal_weight")

    manifest, files = inspect_archive(
        create_export_archive(
            result,
            DashboardSettings(),
            asset_prices,
            benchmark_prices,
        )
    )

    assert manifest["analysis_type"] == "parameter_sensitivity"
    assert len(manifest["evaluation_windows"]) == 3

    for parameter in result.tables:
        assert f"results/{parameter}.csv" in files
        assert (
            manifest["evaluation_windows"][parameter]["observations"]
            == result.tables[parameter].attrs["observations"]
        )


def test_invalid_settings_rejected(sample_market_prices) -> None:
    asset_prices, benchmark_prices = sample_market_prices

    result = run_sample_analysis(asset_prices, benchmark_prices)

    with pytest.raises(TypeError, match="DashboardSettings"):
        create_export_archive(
            result,
            None,
            asset_prices,
            benchmark_prices,
        )
