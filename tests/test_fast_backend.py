"""Compatibility contract for the optional Rust holding kernel."""

import importlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.fast_backend import _python_holding, simulate_holding_period
from src.optimization import maximum_sharpe_weights


@pytest.fixture
def example():
    rng = np.random.default_rng(123)
    returns = pd.DataFrame(
        rng.normal(0.0003, 0.01, (120, 8)),
        index=pd.bdate_range("2023-01-01", periods=120),
        columns=list("ABCDEFGH"),
    )
    weights = pd.Series(np.arange(1.0, 9.0), index=returns.columns)
    weights /= weights.sum()
    return weights, returns


def test_python_path_matches_reference(example, monkeypatch):
    monkeypatch.setenv("QUANTALLOC_ENGINE", "python")
    weights, returns = example
    expected = _python_holding(weights, returns)
    actual = simulate_holding_period(weights, returns)
    pd.testing.assert_series_equal(actual[0], expected[0])
    pd.testing.assert_series_equal(actual[1], expected[1])


def test_auto_falls_back_when_extension_is_absent(example, monkeypatch):
    monkeypatch.setenv("QUANTALLOC_ENGINE", "auto")
    weights, returns = example

    def missing(_name):
        raise ImportError("compiled extension unavailable")

    monkeypatch.setattr(importlib, "import_module", missing)
    actual = simulate_holding_period(weights, returns)
    reference = _python_holding(weights, returns)
    pd.testing.assert_series_equal(actual[0], reference[0])
    pd.testing.assert_series_equal(actual[1], reference[1])


def test_forced_native_is_fail_closed_when_extension_absent(example, monkeypatch):
    monkeypatch.setenv("QUANTALLOC_ENGINE", "rust")

    def missing(_name):
        raise ImportError("compiled extension unavailable")

    monkeypatch.setattr(importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match="requires the quantalloc-fast"):
        simulate_holding_period(*example)


def test_invalid_engine_is_rejected(example, monkeypatch):
    monkeypatch.setenv("QUANTALLOC_ENGINE", "fastest")
    with pytest.raises(ValueError, match="QUANTALLOC_ENGINE"):
        simulate_holding_period(*example)


def test_native_dispatch_preserves_labels_and_drift(example, monkeypatch):
    monkeypatch.setenv("QUANTALLOC_ENGINE", "rust")
    weights, returns = example
    called = []

    def simulated(prices, allocations):
        called.append((prices.shape, allocations.shape))
        # A stand-in for the native ABI, used without a Rust compiler.
        balance = allocations.copy()
        wealth = 1.0
        daily = []
        for row in prices:
            balance *= 1.0 + row
            new_wealth = balance.sum()
            daily.append(new_wealth / wealth - 1.0)
            wealth = new_wealth
        return daily, (balance / wealth).tolist()

    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda _: SimpleNamespace(simulate_holding=simulated),
    )
    gross, ending = simulate_holding_period(weights, returns)
    original_gross, original_ending = _python_holding(weights, returns)
    assert called == [((120, 8), (8,))]
    assert gross.name == "gross_return"
    assert ending.name == "pre_rebalance_weight"
    pd.testing.assert_index_equal(gross.index, returns.index)
    pd.testing.assert_index_equal(ending.index, returns.columns)
    np.testing.assert_allclose(gross, original_gross, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(ending, original_ending, rtol=1e-10, atol=1e-12)


def test_compiled_native_agrees_with_python_when_available(example, monkeypatch):
    pytest.importorskip("_quantalloc_rust")
    weights, returns = example
    monkeypatch.setenv("QUANTALLOC_ENGINE", "rust")
    native_gross, native_weights = simulate_holding_period(weights, returns)
    expected_gross, expected_weights = _python_holding(weights, returns)
    np.testing.assert_allclose(native_gross, expected_gross, atol=1e-12, rtol=1e-10)
    np.testing.assert_allclose(
        native_weights,
        expected_weights,
        atol=1e-12,
        rtol=1e-10,
    )


def test_native_total_loss_of_one_asset_when_available(monkeypatch):
    pytest.importorskip("_quantalloc_rust")
    monkeypatch.setenv("QUANTALLOC_ENGINE", "rust")
    returns = pd.DataFrame(
        {"A": [-1.0, 0.0], "B": [0.0, 0.0]},
        index=pd.bdate_range("2024-01-01", periods=2),
    )
    weights = pd.Series({"A": 0.5, "B": 0.5})
    gross, ending = simulate_holding_period(weights, returns)
    np.testing.assert_allclose(gross, [-0.5, 0.0])
    np.testing.assert_allclose(ending, [0.0, 1.0])


@pytest.mark.parametrize("risk_free_rate", [0.0, 0.025])
def test_maximum_sharpe_gradient_matches_finite_differences(
    monkeypatch, risk_free_rate
):
    import src.optimization as module

    x = np.array([0.3, 0.4, 0.3])
    means = pd.Series([0.11, 0.15, 0.09], index=list("ABC"))
    covariance = pd.DataFrame(
        [[0.04, 0.01, 0.00], [0.01, 0.09, 0.01], [0.00, 0.01, 0.16]],
        index=means.index,
        columns=means.index,
    )
    solve = module._solve

    def checked_solve(objective, asset_count, constraints, **kwargs):
        assert asset_count == 3
        jac = kwargs["gradient"]
        step = 1e-6
        finite = np.array(
            [
                (
                    objective(x + step * np.eye(3)[i])
                    - objective(x - step * np.eye(3)[i])
                )
                / (2 * step)
                for i in range(3)
            ]
        )
        np.testing.assert_allclose(jac(x), finite, rtol=1e-5, atol=1e-7)
        return solve(objective, asset_count, constraints, **kwargs)

    monkeypatch.setattr(module, "_solve", checked_solve)
    result = maximum_sharpe_weights(means, covariance, risk_free_rate)
    assert result.sum() == pytest.approx(1.0)
