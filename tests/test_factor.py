"""Tests for the single-factor market regression."""

import numpy as np
import pandas as pd
import pytest

from src.factor import market_factor_exposure


@pytest.fixture
def factor_returns():
    market = np.array([-0.02, -0.01, 0.0, 0.01, 0.02, 0.03])
    return pd.DataFrame(
        {
            "Levered": 0.001 + 1.5 * market,
            "Flat": np.full(len(market), 0.002),
            "TOPIX": market,
        },
        index=pd.bdate_range("2024-01-01", periods=len(market)),
    )


def test_exact_linear_exposure(factor_returns) -> None:
    result = market_factor_exposure(factor_returns)

    assert result.index.tolist() == ["Levered", "Flat", "TOPIX"]

    assert result.loc["Levered", "beta"] == pytest.approx(1.5)

    assert result.loc["Levered", "alpha_annual"] == pytest.approx(0.001 * 252)

    assert result.loc["Levered", "r_squared"] == pytest.approx(1.0)

    assert result.loc["TOPIX", "beta"] == pytest.approx(1.0)

    assert result.loc["TOPIX", "alpha_annual"] == pytest.approx(0.0, abs=1e-12)

    assert result.loc["TOPIX", "r_squared"] == pytest.approx(1.0)


def test_constant_strategy_has_undefined_r_squared(
    factor_returns,
) -> None:
    result = market_factor_exposure(factor_returns)

    assert result.loc["Flat", "beta"] == pytest.approx(0.0, abs=1e-12)

    assert result.loc["Flat", "alpha_annual"] == pytest.approx(0.002 * 252)

    assert pd.isna(result.loc["Flat", "r_squared"])


def test_constant_benchmark_returns_nan(
    factor_returns,
) -> None:
    factor_returns["TOPIX"] = 0.01

    result = market_factor_exposure(factor_returns)

    assert result.isna().all().all()


def test_annualization_parameter(factor_returns) -> None:
    result = market_factor_exposure(factor_returns, periods_per_year=365)

    assert result.loc["Levered", "alpha_annual"] == pytest.approx(0.001 * 365)


def test_does_not_modify_returns(factor_returns) -> None:
    original = factor_returns.copy(deep=True)

    market_factor_exposure(factor_returns)

    pd.testing.assert_frame_equal(factor_returns, original)


@pytest.mark.parametrize("periods", [0, -1, 252.5, True])
def test_rejects_invalid_annualization(factor_returns, periods) -> None:
    with pytest.raises(ValueError, match="periods_per_year"):
        market_factor_exposure(
            factor_returns,
            periods_per_year=periods,
        )


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -1.5])
def test_rejects_invalid_returns(factor_returns, bad_value) -> None:
    factor_returns.iloc[0, 0] = bad_value

    with pytest.raises(ValueError):
        market_factor_exposure(factor_returns)


def test_rejects_missing_benchmark(
    factor_returns,
) -> None:
    with pytest.raises(ValueError, match="benchmark"):
        market_factor_exposure(factor_returns.drop(columns="TOPIX"))


def test_rejects_non_datetime_index(
    factor_returns,
) -> None:
    factor_returns.index = range(len(factor_returns))

    with pytest.raises(TypeError, match="DatetimeIndex"):
        market_factor_exposure(factor_returns)


def test_rejects_duplicate_dates(factor_returns) -> None:
    factor_returns.index = pd.DatetimeIndex(
        [factor_returns.index[0]] * len(factor_returns)
    )

    with pytest.raises(ValueError, match="sorted and unique"):
        market_factor_exposure(factor_returns)
