"""Process parallelism is opt-in and preserves sensitivity output semantics."""

import pandas as pd
import pytest

import src.sensitivity as sensitivity
from src.portfolio import calculate_asset_returns


@pytest.fixture
def returns(sample_market_prices):
    return calculate_asset_returns(sample_market_prices[0])


@pytest.mark.parametrize("strategy", ["minimum_variance", "hierarchical_risk_parity"])
def test_parallel_sensitivity_matches_serial(returns, strategy) -> None:
    params = {
        "estimation_windows": (252, 504),
        "holding_periods": (21, 63),
        "transaction_cost_rates": (0.0, 0.001, 0.003),
    }
    serial = sensitivity.run_parameter_sensitivity(
        returns, strategy, max_workers=1, **params
    )
    parallel = sensitivity.run_parameter_sensitivity(
        returns, strategy, max_workers=2, **params
    )
    assert serial.strategy == parallel.strategy
    for key, expected in serial.tables.items():
        actual = parallel.tables[key]
        assert actual.attrs == expected.attrs
        pd.testing.assert_frame_equal(
            actual, expected, check_exact=False, rtol=1e-9, atol=1e-11
        )


@pytest.mark.parametrize("workers", [0, -1, 5, True, 1.5, "2"])
def test_parallel_rejects_bad_worker_counts(returns, workers) -> None:
    with pytest.raises(ValueError, match="max_workers"):
        sensitivity.run_parameter_sensitivity(
            returns, "equal_weight", max_workers=workers
        )


def test_one_unique_scenario_does_not_start_processes(returns, monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("unnecessary process pool")

    monkeypatch.setattr(sensitivity, "ProcessPoolExecutor", forbidden)
    result = sensitivity.run_parameter_sensitivity(
        returns,
        "equal_weight",
        estimation_windows=(504,),
        holding_periods=(63,),
        transaction_cost_rates=(0.0, 0.001),
        max_workers=4,
    )
    assert (result.tables["transaction_cost_rate"]["status"] == "OK").all()
