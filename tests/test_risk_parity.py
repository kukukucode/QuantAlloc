"""Tests for equal risk contribution portfolio optimization."""

import numpy as np
import pandas as pd
import pytest

from src.risk import risk_contribution
from src.risk_parity import equal_risk_contribution_weights


def test_erc_weights_equalize_risk_contributions() -> None:
    covariance = pd.DataFrame(
        [
            [0.040, 0.006, 0.004],
            [0.006, 0.090, 0.009],
            [0.004, 0.009, 0.160],
        ],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )

    weights = equal_risk_contribution_weights(covariance)
    contributions = risk_contribution(
        weights,
        covariance,
    )["risk_contribution_pct"]

    assert contributions.to_numpy() == pytest.approx(
        np.full(3, 1.0 / 3.0),
        abs=1e-5,
    )


def test_erc_weights_are_long_only_and_fully_invested() -> None:
    covariance = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.16]],
        index=["Low Vol", "High Vol"],
        columns=["Low Vol", "High Vol"],
    )

    weights = equal_risk_contribution_weights(covariance)

    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= 0.0).all()
    assert (weights <= 1.0).all()
    assert weights["Low Vol"] > weights["High Vol"]
    assert weights.name == "risk_parity"


def test_identical_uncorrelated_assets_receive_equal_weights() -> None:
    covariance = pd.DataFrame(
        np.eye(4) * 0.04,
        index=list("ABCD"),
        columns=list("ABCD"),
    )

    weights = equal_risk_contribution_weights(covariance)

    assert weights.to_numpy() == pytest.approx(np.full(4, 0.25))


@pytest.mark.parametrize(
    "covariance",
    [
        pd.DataFrame(),
        pd.DataFrame([[0.04, 0.01]], index=["A"], columns=["A", "B"]),
        pd.DataFrame(
            [[0.04, 0.02], [0.01, 0.09]],
            index=["A", "B"],
            columns=["A", "B"],
        ),
    ],
)
def test_erc_rejects_invalid_covariance(covariance: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        equal_risk_contribution_weights(covariance)


def test_erc_rejects_zero_variance_asset() -> None:
    covariance = pd.DataFrame(
        [[0.04, 0.0], [0.0, 0.0]],
        index=["A", "B"],
        columns=["A", "B"],
    )

    with pytest.raises(ValueError, match="positive asset variances"):
        equal_risk_contribution_weights(covariance)
