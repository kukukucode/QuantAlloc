"""Offline numeric and PNG-output checks for visualization."""

from matplotlib.figure import Figure
from matplotlib.image import imread
import numpy as np
import pandas as pd
import pytest

from src.visualization import (
    cumulative_return_paths,
    drawdown_paths,
    plot_cumulative_returns,
    plot_drawdowns,
    plot_weight_history,
)


@pytest.fixture
def strategy_returns() -> dict[str, pd.Series]:
    dates = pd.bdate_range("2024-01-01", periods=5)
    return {
        "Strategy": pd.Series([0.5, -0.1, 0.2, -0.1, 0.0], index=dates),
        "TOPIX": pd.Series([0.1, 0.0, -0.1, 0.1], index=dates[1:]),
    }


def test_cumulative_paths_align_and_include_first_common_return(strategy_returns) -> None:
    originals = {name: returns.copy() for name, returns in strategy_returns.items()}
    result = cumulative_return_paths(strategy_returns)
    assert result.index.equals(strategy_returns["TOPIX"].index)
    assert result["Strategy"].to_list() == pytest.approx([-0.1, 0.08, -0.028, -0.028])
    assert result["TOPIX"].to_list() == pytest.approx([0.1, 0.1, -0.01, 0.089])
    for name in originals:
        pd.testing.assert_series_equal(strategy_returns[name], originals[name])


def test_drawdown_paths_include_initial_loss_and_reset_after_recovery(strategy_returns) -> None:
    result = drawdown_paths(strategy_returns)
    assert result["Strategy"].to_list() == pytest.approx([-0.1, 0.0, -0.1, -0.1])
    assert result["TOPIX"].to_list() == pytest.approx([0.0, 0.0, -0.1, -0.01])


@pytest.mark.parametrize("calculate", [cumulative_return_paths, drawdown_paths])
def test_return_paths_handle_total_loss(calculate) -> None:
    returns = pd.Series([-1.0, 0.0], index=pd.bdate_range("2024-01-01", periods=2))
    assert calculate({"Loss": returns})["Loss"].to_list() == [-1.0, -1.0]


@pytest.mark.parametrize("calculate", [cumulative_return_paths, drawdown_paths])
def test_return_paths_reject_internal_missing_dates(strategy_returns, calculate) -> None:
    strategy_returns["Strategy"] = strategy_returns["Strategy"].drop(
        strategy_returns["TOPIX"].index[1]
    )
    with pytest.raises(ValueError, match="missing dates"):
        calculate(strategy_returns)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -1.01, "invalid"])
def test_return_paths_reject_invalid_returns(strategy_returns, invalid) -> None:
    strategy_returns["Strategy"] = strategy_returns["Strategy"].astype(object)
    strategy_returns["Strategy"].iloc[2] = invalid
    with pytest.raises(ValueError):
        cumulative_return_paths(strategy_returns)


def test_return_paths_reject_disjoint_periods(strategy_returns) -> None:
    strategy_returns["TOPIX"].index = pd.bdate_range("2025-01-01", periods=4)
    with pytest.raises(ValueError, match="at least two"):
        drawdown_paths(strategy_returns)


@pytest.mark.parametrize("plot", [plot_cumulative_returns, plot_drawdowns])
def test_return_charts_save_real_pngs_and_create_parents(strategy_returns, tmp_path, plot) -> None:
    output = tmp_path / "nested" / "chart.png"
    assert plot(strategy_returns, output) == output
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    pixels = imread(output)
    assert pixels.shape[0] > 500
    assert pixels.shape[1] > 1000
    assert pixels.std() > 0.01


def test_return_chart_plots_calculated_data_and_labels(strategy_returns, monkeypatch, tmp_path) -> None:
    original_save = Figure.savefig
    recorded = {}

    def record_save(figure, *args, **kwargs):
        axis = figure.axes[0]
        recorded["labels"] = [line.get_label() for line in axis.lines[:2]]
        recorded["values"] = axis.lines[0].get_ydata().copy()
        recorded["ylabel"] = axis.get_ylabel()
        return original_save(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", record_save)
    plot_cumulative_returns(strategy_returns, tmp_path / "chart.png")
    assert recorded["labels"] == ["Strategy", "TOPIX"]
    assert recorded["values"] == pytest.approx([-0.1, 0.08, -0.028, -0.028])
    assert recorded["ylabel"] == "Cumulative return"


@pytest.fixture
def target_weights() -> pd.DataFrame:
    return pd.DataFrame(
        [[0.4, 0.6], [0.7, 0.3]],
        index=pd.to_datetime(["2024-01-01", "2024-04-01"]), columns=["A", "B"],
    )


@pytest.mark.parametrize("single_rebalance", [False, True])
def test_weight_heatmap_saves_png_without_mutating_input(target_weights, tmp_path, single_rebalance) -> None:
    if single_rebalance:
        target_weights = target_weights.iloc[:1]
    original = target_weights.copy(deep=True)
    output = tmp_path / "weights.png"
    assert plot_weight_history(target_weights, str(output)) == output
    assert imread(output).std() > 0.01
    pd.testing.assert_frame_equal(target_weights, original)


def test_weight_heatmap_sorts_dates_and_preserves_asset_labels(target_weights, monkeypatch, tmp_path) -> None:
    original_save = Figure.savefig
    recorded = {}

    def record_save(figure, *args, **kwargs):
        axis = figure.axes[0]
        recorded["values"] = axis.images[0].get_array().copy()
        recorded["labels"] = [label.get_text() for label in axis.get_yticklabels()]
        return original_save(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", record_save)
    plot_weight_history(target_weights.iloc[::-1], tmp_path / "weights.png")
    np.testing.assert_allclose(recorded["values"], target_weights.to_numpy().T)
    assert recorded["labels"] == ["A", "B"]


@pytest.mark.parametrize("invalid_row", [[0.4, 0.4], [-0.1, 1.1], [np.nan, 1.0], [np.inf, 0.0], ["bad", 0.5]])
def test_weight_heatmap_rejects_invalid_weights(target_weights, tmp_path, invalid_row) -> None:
    target_weights = target_weights.astype(object)
    target_weights.iloc[0] = invalid_row
    with pytest.raises(ValueError):
        plot_weight_history(target_weights, tmp_path / "weights.png")
    assert not (tmp_path / "weights.png").exists()


@pytest.mark.parametrize("problem", ["empty", "date_type", "duplicate_dates", "nat", "duplicate_assets"])
def test_weight_heatmap_rejects_invalid_structure(target_weights, tmp_path, problem) -> None:
    if problem == "empty":
        target_weights = pd.DataFrame()
    elif problem == "date_type":
        target_weights.index = [0, 1]
    elif problem == "duplicate_dates":
        target_weights.index = target_weights.index[[0, 0]]
    elif problem == "nat":
        target_weights.index = pd.DatetimeIndex([pd.NaT, target_weights.index[1]])
    else:
        target_weights.columns = ["A", "A"]
    with pytest.raises((ValueError, TypeError)):
        plot_weight_history(target_weights, tmp_path / "weights.png")


def test_png_extension_is_required(strategy_returns, tmp_path) -> None:
    with pytest.raises(ValueError, match=".png extension"):
        plot_drawdowns(strategy_returns, tmp_path / "chart.svg")
    assert not (tmp_path / "chart.svg").exists()


def test_figure_is_cleared_if_saving_fails(strategy_returns, monkeypatch, tmp_path) -> None:
    figures = []

    def fail_save(figure, *args, **kwargs):
        figures.append(figure)
        raise OSError("disk unavailable")

    monkeypatch.setattr(Figure, "savefig", fail_save)
    with pytest.raises(OSError, match="disk unavailable"):
        plot_drawdowns(strategy_returns, tmp_path / "chart.png")
    assert not figures[0].axes
