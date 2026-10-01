"""Tests for BL allocation in the rolling, cost-aware backtest engine."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.backtest import BlackLittermanConfig, walk_forward_backtest
from src.covariance import estimate_covariance


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
def static_config() -> BlackLittermanConfig:
    return BlackLittermanConfig(
        market_weights=pd.Series({"B": 0.4, "A": 0.6}),
        view_matrix=pd.DataFrame(
            [[-1.0, 1.0]], index=["relative"], columns=["B", "A"]
        ),
        view_returns=pd.Series({"relative": 0.02}),
        view_uncertainty=pd.DataFrame(
            [[0.002]], index=["relative"], columns=["relative"]
        ),
    )


def _neutral_config(
    training: pd.DataFrame,
    covariance_method: str = "sample",
    risk_free_rate: float = 0.0,
) -> BlackLittermanConfig:
    """Neutral views preserve the known 60/40 market tangency portfolio."""
    covariance = estimate_covariance(training, method=covariance_method)
    market_weights = pd.Series({"B": 0.4, "A": 0.6})
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


@pytest.mark.parametrize("covariance_method", ["sample", "ledoit_wolf"])
@pytest.mark.parametrize("risk_free_rate", [0.0, 0.03])
def test_provider_receives_only_rolling_training_windows(
    asset_returns: pd.DataFrame,
    covariance_method: str,
    risk_free_rate: float,
) -> None:
    seen: list[pd.DataFrame] = []

    def provider(training: pd.DataFrame) -> BlackLittermanConfig:
        seen.append(training.copy())
        return _neutral_config(training, covariance_method, risk_free_rate)

    result = walk_forward_backtest(
        asset_returns,
        "black_litterman",
        estimation_window=4,
        holding_period=2,
        risk_free_rate=risk_free_rate,
        covariance_method=covariance_method,
        black_litterman=provider,
    )

    assert len(seen) == len(result.periods) == 2
    for position, training in enumerate(seen):
        start = position * 2
        pd.testing.assert_frame_equal(training, asset_returns.iloc[start:start + 4])
        assert training.index[-1] < result.periods.iloc[position]["test_start"]
    expected = pd.DataFrame(
        [[0.6, 0.4], [0.6, 0.4]],
        columns=["A", "B"],
        index=pd.DatetimeIndex(asset_returns.index[[4, 6]], name="rebalance"),
    )
    pd.testing.assert_frame_equal(result.target_weights, expected, atol=1e-5, rtol=0.0)
    assert result.target_weights.sum(axis=1).to_list() == pytest.approx([1.0, 1.0])
    assert (result.target_weights >= 0.0).all().all()
    assert (result.target_weights <= 1.0).all().all()
    assert result.net_returns.index.equals(asset_returns.index[4:])


def test_static_config_reestimates_prior_and_covariance_each_period(
    asset_returns: pd.DataFrame,
    static_config: BlackLittermanConfig,
) -> None:
    result = walk_forward_backtest(
        asset_returns,
        "black_litterman",
        estimation_window=4,
        holding_period=2,
        risk_free_rate=0.03,
        black_litterman=static_config,
    )

    for period in range(2):
        sigma = asset_returns.iloc[period * 2:period * 2 + 4].cov().to_numpy() * 252
        prior = 2.5 * sigma @ np.array([0.6, 0.4]) + 0.03
        p = np.array([1.0, -1.0])
        # Independent conditioning formula for one relative view.
        posterior = prior + 0.05 * sigma @ p * (
            (0.02 - p @ prior) / (0.002 + 0.05 * p @ sigma @ p)
        )
        tangency = np.linalg.solve(sigma, posterior - 0.03)
        assert (tangency > 0.0).all()
        expected = tangency / tangency.sum()
        assert result.target_weights.iloc[period].to_numpy() == pytest.approx(
            expected, abs=1e-5
        )
    assert not np.allclose(result.target_weights.iloc[0], result.target_weights.iloc[1])


@pytest.mark.parametrize("covariance_method", ["sample", "ledoit_wolf"])
def test_future_returns_do_not_change_first_bl_weights(
    asset_returns: pd.DataFrame,
    static_config: BlackLittermanConfig,
    covariance_method: str,
) -> None:
    changed = asset_returns.copy()
    changed.iloc[4:] = [0.4, -0.3]
    results = [
        walk_forward_backtest(
            data,
            "black_litterman",
            estimation_window=4,
            holding_period=2,
            covariance_method=covariance_method,
            black_litterman=static_config,
        )
        for data in (asset_returns, changed)
    ]

    pd.testing.assert_series_equal(
        results[0].target_weights.iloc[0], results[1].target_weights.iloc[0]
    )


@pytest.mark.parametrize("cost_rate", [0.0, 0.01])
def test_bl_returns_turnover_and_costs_match_buy_and_hold(
    asset_returns: pd.DataFrame,
    cost_rate: float,
) -> None:
    result = walk_forward_backtest(
        asset_returns,
        "black_litterman",
        estimation_window=4,
        holding_period=2,
        transaction_cost_rate=cost_rate,
        black_litterman=_neutral_config,
    )

    drifted = np.array([0.6 * 1.10, 0.4]) / 1.06
    turnover = np.abs(np.array([0.6, 0.4]) - drifted).sum()
    assert result.pre_rebalance_weights.iloc[1].to_numpy() == pytest.approx(
        drifted, abs=1e-5
    )
    assert result.turnover.to_list() == pytest.approx([0.0, turnover], abs=1e-5)
    assert result.transaction_costs.to_list() == pytest.approx(
        [0.0, turnover * cost_rate], abs=1e-7
    )
    assert result.gross_returns.to_list() == pytest.approx(
        [0.06, 0.0, 0.014, 0.0], abs=1e-6
    )
    assert result.net_returns.to_list() == pytest.approx(
        [0.06, 0.0, (1.014 * (1.0 - turnover * cost_rate) - 1.0), 0.0],
        abs=1e-6,
    )
    if cost_rate == 0.0:
        pd.testing.assert_series_equal(result.gross_returns, result.net_returns)


def test_provider_cannot_mutate_engine_data(asset_returns: pd.DataFrame) -> None:
    original = asset_returns.copy(deep=True)

    def mutating_provider(training: pd.DataFrame) -> BlackLittermanConfig:
        config = _neutral_config(training)
        training.iloc[:, :] = 99.0
        return config

    result = walk_forward_backtest(
        asset_returns,
        "black_litterman",
        estimation_window=4,
        holding_period=2,
        black_litterman=mutating_provider,
    )

    pd.testing.assert_frame_equal(asset_returns, original)
    assert result.gross_returns.to_list() == pytest.approx(
        [0.06, 0.0, 0.014, 0.0], abs=1e-6
    )


@pytest.mark.parametrize("config", [None, {}])
def test_bl_requires_explicit_inputs(asset_returns: pd.DataFrame, config: object) -> None:
    with pytest.raises(ValueError, match="requires a BlackLittermanConfig"):
        walk_forward_backtest(
            asset_returns, "black_litterman", estimation_window=4,
            holding_period=2, black_litterman=config,  # type: ignore[arg-type]
        )


def test_invalid_provider_result_is_rejected(asset_returns: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="must resolve to a BlackLittermanConfig"):
        walk_forward_backtest(
            asset_returns, "black_litterman", estimation_window=4,
            holding_period=2, black_litterman=lambda training: None,
        )


def test_bl_inputs_are_not_silently_ignored(
    asset_returns: pd.DataFrame, static_config: BlackLittermanConfig,
) -> None:
    with pytest.raises(ValueError, match="BL inputs require strategy"):
        walk_forward_backtest(
            asset_returns, "equal_weight", estimation_window=4,
            holding_period=2, black_litterman=static_config,
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"tau": 0.0}, "tau must be a positive finite number"),
        ({"market_weights": pd.Series({"A": 0.6, "C": 0.4})}, "assets must match"),
    ],
)
def test_invalid_bl_config_fields_are_rejected(
    asset_returns: pd.DataFrame,
    static_config: BlackLittermanConfig,
    changes: dict,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        walk_forward_backtest(
            asset_returns, "black_litterman", estimation_window=4,
            holding_period=2, black_litterman=replace(static_config, **changes),
        )


def test_singular_covariance_does_not_fall_back_to_another_strategy(
    asset_returns: pd.DataFrame, static_config: BlackLittermanConfig,
) -> None:
    singular = asset_returns.copy()
    singular.iloc[:4] = 0.0
    with pytest.raises(ValueError, match="must be invertible"):
        walk_forward_backtest(
            singular, "black_litterman", estimation_window=4,
            holding_period=2, black_litterman=static_config,
        )
