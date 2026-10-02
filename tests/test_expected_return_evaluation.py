"""End-to-end tests for historical-mean versus BL OOS comparison."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import BlackLittermanConfig
from src.covariance import estimate_covariance
from src.evaluation import compare_expected_return_models, performance_summary


@pytest.fixture
def asset_returns() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "A": [0.01, -0.01, 0.0, 0.0, 0.10, 0.0, 0.01, 0.0],
            "B": [0.0, 0.0, 0.02, -0.02, 0.0, 0.0, 0.02, 0.0],
        },
        index=pd.bdate_range("2024-01-01", periods=8),
    )


@pytest.fixture
def benchmark(asset_returns: pd.DataFrame) -> pd.Series:
    return pd.Series(
        [0.0, 0.0, 0.0, 0.0, 0.02, -0.01, 0.03, -0.02],
        index=asset_returns.index,
        name="1306.T",
    )


def _neutral_config(
    training: pd.DataFrame,
    covariance_method: str = "sample",
    risk_free_rate: float = 0.0,
) -> BlackLittermanConfig:
    covariance = estimate_covariance(training, method=covariance_method)
    market_weights = pd.Series({"A": 0.6, "B": 0.4})
    total_prior = 2.5 * covariance.dot(market_weights) + risk_free_rate
    views = ["view_A", "view_B"]
    return BlackLittermanConfig(
        market_weights=market_weights,
        view_matrix=pd.DataFrame(np.eye(2), index=views, columns=["A", "B"]),
        view_returns=pd.Series(total_prior.to_numpy(), index=views),
        view_uncertainty=pd.DataFrame(
            np.eye(2) * 0.002, index=views, columns=views
        ),
    )


def _expected_bl_net_returns(index: pd.DatetimeIndex, cost_rate: float) -> pd.Series:
    drifted = np.array([0.66, 0.4]) / 1.06
    turnover = np.abs(np.array([0.6, 0.4]) - drifted).sum()
    return pd.Series(
        [0.06, 0.0, 1.014 * (1.0 - turnover * cost_rate) - 1.0, 0.0],
        index=index,
    )


def _assert_net_metrics(
    row: pd.Series, returns: pd.Series, risk_free_rate: float = 0.0
) -> None:
    expected = performance_summary(returns, risk_free_rate)
    for metric, value in expected.items():
        assert row[f"net_{metric}"] == pytest.approx(value, rel=1e-4, abs=1e-6)


@pytest.mark.parametrize("covariance_method", ["sample", "ledoit_wolf"])
def test_comparison_matches_known_oos_returns_turnover_and_costs(
    asset_returns: pd.DataFrame,
    benchmark: pd.Series,
    covariance_method: str,
) -> None:
    result = compare_expected_return_models(
        asset_returns,
        lambda training: _neutral_config(training, covariance_method),
        benchmark_returns=benchmark,
        estimation_window=4,
        holding_period=2,
        transaction_cost_rate=0.01,
        covariance_method=covariance_method,
    )

    assert result.index.to_list() == ["historical_mean", "black_litterman", "TOPIX"]
    assert result.index.name == "expected_return_model"
    assert result.columns.to_list() == [
        "net_cagr", "net_volatility", "net_sharpe", "net_max_drawdown",
        "average_target_weight_change", "average_turnover", "total_turnover",
        "maximum_turnover", "total_transaction_cost",
    ]
    dates = asset_returns.index[4:]
    _assert_net_metrics(
        result.loc["historical_mean"], pd.Series([0.10, 0.0, 0.01, 0.0], index=dates)
    )
    _assert_net_metrics(result.loc["black_litterman"], _expected_bl_net_returns(dates, 0.01))
    _assert_net_metrics(result.loc["TOPIX"], benchmark.loc[dates])
    turnover = np.abs(np.array([0.6, 0.4]) - np.array([0.66, 0.4]) / 1.06).sum()
    bl = result.loc["black_litterman"]
    assert bl["total_turnover"] == pytest.approx(turnover, abs=1e-5)
    assert bl["average_turnover"] == pytest.approx(turnover / 2, abs=1e-5)
    assert bl["maximum_turnover"] == pytest.approx(turnover, abs=1e-5)
    assert bl["total_transaction_cost"] == pytest.approx(turnover * 0.01, abs=1e-7)
    assert bl["average_target_weight_change"] == pytest.approx(0.0, abs=1e-5)
    assert result.loc["historical_mean", "total_transaction_cost"] == pytest.approx(0.0)
    assert result.loc["TOPIX"].iloc[4:].isna().all()
    assert result.attrs == {
        "start_date": dates[0], "end_date": dates[-1], "observations": 4,
        "risk_free_rate": 0.0, "transaction_cost_rate": 0.01,
        "estimation_window": 4, "holding_period": 2,
        "covariance_method": covariance_method, "optimizer": "maximum_sharpe",
    }


def test_comparison_trims_all_metrics_and_trades_to_common_benchmark_period(
    asset_returns: pd.DataFrame, benchmark: pd.Series,
) -> None:
    # Begin inside the first holding window and end inside the second.
    short_benchmark = benchmark.iloc[5:7]
    result = compare_expected_return_models(
        asset_returns, _neutral_config, benchmark_returns=short_benchmark,
        estimation_window=4, holding_period=2, transaction_cost_rate=0.01,
    )
    dates = short_benchmark.index
    _assert_net_metrics(
        result.loc["historical_mean"], pd.Series([0.0, 0.01], index=dates)
    )
    _assert_net_metrics(
        result.loc["black_litterman"],
        _expected_bl_net_returns(asset_returns.index[4:], 0.01).loc[dates],
    )
    _assert_net_metrics(result.loc["TOPIX"], short_benchmark)
    bl = result.loc["black_litterman"]
    assert bl["average_turnover"] == pytest.approx(bl["total_turnover"])
    assert bl["total_turnover"] > 0.0
    assert result.attrs["start_date"] == dates[0]
    assert result.attrs["end_date"] == dates[-1]
    assert result.attrs["observations"] == 2


def test_comparison_passes_training_only_to_bl_provider(
    asset_returns: pd.DataFrame,
) -> None:
    seen: list[pd.DataFrame] = []

    def provider(training: pd.DataFrame) -> BlackLittermanConfig:
        seen.append(training.copy())
        return _neutral_config(training)

    result = compare_expected_return_models(
        asset_returns, provider, estimation_window=4, holding_period=2
    )

    assert len(seen) == 2
    pd.testing.assert_frame_equal(seen[0], asset_returns.iloc[:4])
    pd.testing.assert_frame_equal(seen[1], asset_returns.iloc[2:6])
    assert result.index.to_list() == ["historical_mean", "black_litterman"]


def test_comparison_applies_risk_free_rate_to_both_optimization_and_metrics(
    asset_returns: pd.DataFrame, benchmark: pd.Series,
) -> None:
    result = compare_expected_return_models(
        asset_returns,
        lambda training: _neutral_config(training, risk_free_rate=0.03),
        benchmark_returns=benchmark,
        estimation_window=4, holding_period=2,
        risk_free_rate=0.03, transaction_cost_rate=0.01,
    )

    dates = asset_returns.index[4:]
    # Negative initial excess means select B (higher standalone Sharpe).
    # Next rebalance selects A: turnover=2 and a 2% transaction charge.
    historical = pd.Series([0.0, 0.0, 1.01 * 0.98 - 1.0, 0.0], index=dates)
    _assert_net_metrics(result.loc["historical_mean"], historical, 0.03)
    _assert_net_metrics(
        result.loc["black_litterman"], _expected_bl_net_returns(dates, 0.01), 0.03
    )
    _assert_net_metrics(result.loc["TOPIX"], benchmark.loc[dates], 0.03)
    assert result.loc["historical_mean", "total_turnover"] == pytest.approx(2.0)
    assert result.attrs["risk_free_rate"] == 0.03


def test_zero_cost_comparison_uses_gross_returns(asset_returns: pd.DataFrame) -> None:
    result = compare_expected_return_models(
        asset_returns, _neutral_config, estimation_window=4,
        holding_period=2, transaction_cost_rate=0.0,
    )

    _assert_net_metrics(
        result.loc["black_litterman"],
        pd.Series([0.06, 0.0, 0.014, 0.0], index=asset_returns.index[4:]),
    )
    assert (result["total_transaction_cost"] == 0.0).all()
    assert result.loc["black_litterman", "total_turnover"] > 0.0


def test_static_bl_configuration_is_supported(asset_returns: pd.DataFrame) -> None:
    config = _neutral_config(asset_returns.iloc[:4])
    result = compare_expected_return_models(
        asset_returns, config, estimation_window=4, holding_period=2
    )

    assert result.index.to_list() == ["historical_mean", "black_litterman"]
    assert np.isfinite(result.to_numpy()).all()


def test_missing_internal_benchmark_date_is_rejected(
    asset_returns: pd.DataFrame, benchmark: pd.Series,
) -> None:
    with pytest.raises(ValueError, match="missing dates"):
        compare_expected_return_models(
            asset_returns, _neutral_config,
            benchmark_returns=benchmark.drop(asset_returns.index[5]),
            estimation_window=4, holding_period=2,
        )


def test_non_overlapping_benchmark_is_rejected(
    asset_returns: pd.DataFrame,
) -> None:
    benchmark = pd.Series([0.01, 0.02], index=pd.bdate_range("2025-01-01", periods=2))
    with pytest.raises(ValueError, match="share at least two OOS dates"):
        compare_expected_return_models(
            asset_returns, _neutral_config, benchmark_returns=benchmark,
            estimation_window=4, holding_period=2,
        )


def test_comparison_requires_bl_configuration(asset_returns: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="requires a BlackLittermanConfig"):
        compare_expected_return_models(
            asset_returns, None, estimation_window=4, holding_period=2,
        )
