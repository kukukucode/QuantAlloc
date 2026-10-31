"""Verify the sensitivity dashboard reuses the original backtester."""

import pytest

from src.dashboard import run_sample_sensitivity
from src.portfolio import calculate_asset_returns
from src.sensitivity import run_parameter_sensitivity


def test_dashboard_matches_pure_engine(
    sample_market_prices,
) -> None:
    asset_prices, _ = sample_market_prices

    dashboard = run_sample_sensitivity(asset_prices, "equal_weight")
    expected = run_parameter_sensitivity(
        calculate_asset_returns(asset_prices),
        "equal_weight",
    )

    for parameter in expected.tables:
        actual_table = dashboard.tables[parameter]
        expected_table = expected.tables[parameter]

        assert actual_table.attrs == expected_table.attrs

        for value in expected_table.index:
            assert (
                actual_table.loc[value, "status"] == expected_table.loc[value, "status"]
            )

            if actual_table.loc[value, "status"] == "OK":
                assert actual_table.loc[value, "cagr"] == pytest.approx(
                    expected_table.loc[value, "cagr"]
                )
