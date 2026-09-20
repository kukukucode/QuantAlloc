"""Tests for core Black-Litterman calculations."""

import numpy as np
import pandas as pd
import pytest

from src.black_litterman import implied_equilibrium_returns, validate_views


@pytest.fixture
def covariance() -> pd.DataFrame:
    return pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.09]],
        index=["A", "B"],
        columns=["A", "B"],
    )


def test_implied_equilibrium_returns_match_formula(
    covariance: pd.DataFrame,
) -> None:
    market_weights = pd.Series({"A": 0.60, "B": 0.40})

    result = implied_equilibrium_returns(
        market_weights,
        covariance,
        risk_aversion=2.5,
    )

    expected = pd.Series(
        {"A": 0.07, "B": 0.105},
        name="implied_equilibrium_return",
    )
    pd.testing.assert_series_equal(result, expected)


def test_market_weights_are_aligned_to_covariance_assets(
    covariance: pd.DataFrame,
) -> None:
    market_weights = pd.Series({"B": 0.40, "A": 0.60})

    result = implied_equilibrium_returns(
        market_weights,
        covariance,
        risk_aversion=2.5,
    )

    assert result.index.to_list() == ["A", "B"]
    assert result.to_list() == pytest.approx([0.07, 0.105])


def test_risk_aversion_scales_implied_returns(
    covariance: pd.DataFrame,
) -> None:
    market_weights = pd.Series({"A": 0.60, "B": 0.40})

    low = implied_equilibrium_returns(
        market_weights,
        covariance,
        risk_aversion=1.0,
    )
    high = implied_equilibrium_returns(
        market_weights,
        covariance,
        risk_aversion=3.0,
    )

    pd.testing.assert_series_equal(high, low * 3.0)


@pytest.mark.parametrize(
    "risk_aversion",
    [0.0, -1.0, np.nan, np.inf, True, "invalid"],
)
def test_invalid_risk_aversion_is_rejected(
    covariance: pd.DataFrame,
    risk_aversion: object,
) -> None:
    market_weights = pd.Series({"A": 0.60, "B": 0.40})

    with pytest.raises(ValueError, match="positive finite number"):
        implied_equilibrium_returns(
            market_weights,
            covariance,
            risk_aversion=risk_aversion,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "market_weights",
    [
        pd.Series({"A": 0.60, "C": 0.40}),
        pd.Series({"A": 0.60, "B": 0.30}),
        pd.Series({"A": 1.10, "B": -0.10}),
        pd.Series({"A": np.nan, "B": np.nan}),
    ],
)
def test_invalid_market_weights_are_rejected(
    covariance: pd.DataFrame,
    market_weights: pd.Series,
) -> None:
    with pytest.raises(ValueError):
        implied_equilibrium_returns(
            market_weights,
            covariance,
            risk_aversion=2.5,
        )


def test_duplicate_market_weight_assets_are_rejected(
    covariance: pd.DataFrame,
) -> None:
    market_weights = pd.Series(
        [0.50, 0.25, 0.25],
        index=["A", "A", "B"],
    )

    with pytest.raises(ValueError, match="index must be unique"):
        implied_equilibrium_returns(
            market_weights,
            covariance,
            risk_aversion=2.5,
        )


@pytest.mark.parametrize("market_weights", [pd.Series(dtype=float), [0.5, 0.5]])
def test_market_weights_must_be_a_non_empty_series(
    covariance: pd.DataFrame,
    market_weights: object,
) -> None:
    with pytest.raises(ValueError, match="non-empty Series"):
        implied_equilibrium_returns(
            market_weights,  # type: ignore[arg-type]
            covariance,
            risk_aversion=2.5,
        )


def test_invalid_covariance_is_rejected() -> None:
    market_weights = pd.Series({"A": 0.50, "B": 0.50})
    covariance = pd.DataFrame(
        [[0.04, 0.02], [0.01, 0.09]],
        index=["A", "B"],
        columns=["A", "B"],
    )

    with pytest.raises(ValueError, match="symmetric"):
        implied_equilibrium_returns(
            market_weights,
            covariance,
            risk_aversion=2.5,
        )


@pytest.fixture
def black_litterman_views() -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
]:
    pick_matrix = pd.DataFrame(
        [[1.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
        index=["relative", "absolute"],
        columns=["A", "B", "C"],
    )
    view_returns = pd.Series(
        {"relative": 0.03, "absolute": 0.05},
    )
    view_uncertainty = pd.DataFrame(
        [[0.02, 0.0], [0.0, 0.01]],
        index=["relative", "absolute"],
        columns=["relative", "absolute"],
    )
    return pick_matrix, view_returns, view_uncertainty


def test_validate_views_aligns_assets_and_view_labels(
    black_litterman_views: tuple[
        pd.DataFrame,
        pd.Series,
        pd.DataFrame,
    ],
) -> None:
    pick_matrix, view_returns, view_uncertainty = black_litterman_views

    validated_p, validated_q, validated_omega = validate_views(
        pick_matrix[["C", "A", "B"]],
        view_returns.reindex(["absolute", "relative"]),
        view_uncertainty.reindex(
            index=["absolute", "relative"],
            columns=["absolute", "relative"],
        ),
        assets=pd.Index(["A", "B", "C"]),
    )

    assert validated_p.columns.to_list() == ["A", "B", "C"]
    assert validated_p.index.to_list() == ["relative", "absolute"]
    assert validated_q.index.to_list() == ["relative", "absolute"]
    assert validated_omega.index.to_list() == ["relative", "absolute"]
    assert validated_omega.columns.to_list() == ["relative", "absolute"]


@pytest.mark.parametrize(
    ("component", "invalid_value", "message"),
    [
        ("pick_matrix", pd.DataFrame(), "non-empty DataFrame"),
        ("view_returns", pd.Series(dtype=float), "non-empty Series"),
        ("view_uncertainty", pd.DataFrame(), "non-empty DataFrame"),
    ],
)
def test_validate_views_rejects_empty_inputs(
    black_litterman_views: tuple[
        pd.DataFrame,
        pd.Series,
        pd.DataFrame,
    ],
    component: str,
    invalid_value: object,
    message: str,
) -> None:
    pick_matrix, view_returns, view_uncertainty = black_litterman_views
    inputs = {
        "pick_matrix": pick_matrix,
        "view_returns": view_returns,
        "view_uncertainty": view_uncertainty,
    }
    inputs[component] = invalid_value

    with pytest.raises(ValueError, match=message):
        validate_views(
            inputs["pick_matrix"],  # type: ignore[arg-type]
            inputs["view_returns"],  # type: ignore[arg-type]
            inputs["view_uncertainty"],  # type: ignore[arg-type]
            assets=pd.Index(["A", "B", "C"]),
        )


def test_validate_views_rejects_asset_mismatch(
    black_litterman_views: tuple[
        pd.DataFrame,
        pd.Series,
        pd.DataFrame,
    ],
) -> None:
    pick_matrix, view_returns, view_uncertainty = black_litterman_views

    with pytest.raises(ValueError, match="columns and assets must match"):
        validate_views(
            pick_matrix,
            view_returns,
            view_uncertainty,
            assets=pd.Index(["A", "B", "D"]),
        )


@pytest.mark.parametrize("component", ["view_returns", "view_uncertainty"])
def test_validate_views_rejects_view_label_mismatch(
    black_litterman_views: tuple[
        pd.DataFrame,
        pd.Series,
        pd.DataFrame,
    ],
    component: str,
) -> None:
    pick_matrix, view_returns, view_uncertainty = black_litterman_views
    if component == "view_returns":
        view_returns = view_returns.rename(index={"absolute": "unknown"})
    else:
        view_uncertainty = view_uncertainty.rename(
            index={"absolute": "unknown"},
            columns={"absolute": "unknown"},
        )

    with pytest.raises(ValueError, match="views must match"):
        validate_views(
            pick_matrix,
            view_returns,
            view_uncertainty,
            assets=pd.Index(["A", "B", "C"]),
        )


@pytest.mark.parametrize(
    ("component", "invalid_value"),
    [
        ("pick_matrix", "invalid"),
        ("pick_matrix", np.nan),
        ("view_returns", "invalid"),
        ("view_returns", np.inf),
        ("view_uncertainty", "invalid"),
        ("view_uncertainty", np.nan),
    ],
)
def test_validate_views_rejects_non_finite_or_non_numeric_values(
    black_litterman_views: tuple[
        pd.DataFrame,
        pd.Series,
        pd.DataFrame,
    ],
    component: str,
    invalid_value: object,
) -> None:
    pick_matrix, view_returns, view_uncertainty = black_litterman_views
    if component == "pick_matrix":
        pick_matrix = pick_matrix.astype(object)
        pick_matrix.loc["relative", "A"] = invalid_value
    elif component == "view_returns":
        view_returns = view_returns.astype(object)
        view_returns.loc["relative"] = invalid_value
    else:
        view_uncertainty = view_uncertainty.astype(object)
        view_uncertainty.loc["relative", "relative"] = invalid_value

    with pytest.raises(ValueError, match="must be numeric|finite values"):
        validate_views(
            pick_matrix,
            view_returns,
            view_uncertainty,
            assets=pd.Index(["A", "B", "C"]),
        )


def test_validate_views_rejects_zero_exposure_view(
    black_litterman_views: tuple[
        pd.DataFrame,
        pd.Series,
        pd.DataFrame,
    ],
) -> None:
    pick_matrix, view_returns, view_uncertainty = black_litterman_views
    pick_matrix = pick_matrix.copy()
    pick_matrix.loc["relative"] = 0.0

    with pytest.raises(ValueError, match="non-zero exposure"):
        validate_views(
            pick_matrix,
            view_returns,
            view_uncertainty,
            assets=pd.Index(["A", "B", "C"]),
        )


@pytest.mark.parametrize(
    ("view_uncertainty", "message"),
    [
        (
            pd.DataFrame(
                [[0.02, 0.01], [0.0, 0.01]],
                index=["relative", "absolute"],
                columns=["relative", "absolute"],
            ),
            "symmetric",
        ),
        (
            pd.DataFrame(
                [[0.02, 0.0], [0.0, 0.0]],
                index=["relative", "absolute"],
                columns=["relative", "absolute"],
            ),
            "positive definite",
        ),
    ],
)
def test_validate_views_rejects_invalid_uncertainty_matrix(
    black_litterman_views: tuple[
        pd.DataFrame,
        pd.Series,
        pd.DataFrame,
    ],
    view_uncertainty: pd.DataFrame,
    message: str,
) -> None:
    pick_matrix, view_returns, _ = black_litterman_views

    with pytest.raises(ValueError, match=message):
        validate_views(
            pick_matrix,
            view_returns,
            view_uncertainty,
            assets=pd.Index(["A", "B", "C"]),
        )
