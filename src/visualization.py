"""Headless PNG charts built from existing returns and target weights."""

from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from src.evaluation import _align_strategy_returns
from src.metrics import wealth_index


def _aligned_wealth(strategy_returns: dict[str, pd.Series]) -> pd.DataFrame:
    """Use the same common-period alignment as performance comparisons."""
    aligned = _align_strategy_returns(strategy_returns)
    return aligned.apply(wealth_index)


def cumulative_return_paths(strategy_returns: dict[str, pd.Series]) -> pd.DataFrame:
    """Compound every return in the shared OOS period, including its first day."""
    return _aligned_wealth(strategy_returns) - 1.0


def drawdown_paths(strategy_returns: dict[str, pd.Series]) -> pd.DataFrame:
    """Calculate drawdowns against running peaks, including initial wealth 1."""
    wealth = _aligned_wealth(strategy_returns)
    return wealth / wealth.cummax().clip(lower=1.0) - 1.0


def _save_figure(figure: Figure, output_path: str | Path) -> Path:
    """Save without pyplot or a GUI, then release the figure's artists."""
    try:
        path = Path(output_path)
        if path.suffix.lower() != ".png":
            raise ValueError("output_path must have a .png extension")
        path.parent.mkdir(parents=True, exist_ok=True)
        FigureCanvasAgg(figure)
        figure.savefig(path, format="png", dpi=150)
        return path
    finally:
        figure.clear()


def _plot_return_paths(
    paths: pd.DataFrame, output_path: str | Path, title: str, ylabel: str
) -> Path:
    figure = Figure(figsize=(11, 6), layout="constrained")
    axis = figure.subplots()
    for name in paths.columns:
        axis.plot(paths.index, paths[name], label=name, linewidth=1.6)
    axis.axhline(0.0, color="grey", linestyle="--", linewidth=0.7)
    axis.set_title(title)
    axis.set_xlabel("Date")
    axis.set_ylabel(ylabel)
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axis.grid(alpha=0.25)
    axis.legend(loc="best", fontsize=9, ncol=2)
    start, end = paths.index[0].date(), paths.index[-1].date()
    figure.suptitle(f"Common OOS period: {start} to {end}", fontsize=10)
    return _save_figure(figure, output_path)


def plot_cumulative_returns(
    strategy_returns: dict[str, pd.Series],
    output_path: str | Path,
    *,
    title: str = "Net Cumulative Returns",
) -> Path:
    """Save common-period cumulative returns; callers supply net or gross data."""
    return _plot_return_paths(
        cumulative_return_paths(strategy_returns), output_path, title,
        "Cumulative return",
    )


def plot_drawdowns(
    strategy_returns: dict[str, pd.Series],
    output_path: str | Path,
    *,
    title: str = "Net Drawdowns",
) -> Path:
    """Save common-period drawdowns without dropping initial losses."""
    return _plot_return_paths(
        drawdown_paths(strategy_returns), output_path, title, "Drawdown",
    )


def plot_weight_history(
    target_weights: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str = "Target Weights by Rebalance",
) -> Path:
    """Save a heatmap of long-only target weights, not drifted daily holdings.

    Each column is one recorded rebalance. Dates are discrete observations,
    not equally spaced calendar intervals. A single rebalance is supported.
    """
    if not isinstance(target_weights, pd.DataFrame) or target_weights.empty:
        raise ValueError("target_weights must be a non-empty DataFrame")
    if not isinstance(target_weights.index, pd.DatetimeIndex):
        raise TypeError("target_weights must use a DatetimeIndex")
    if target_weights.index.has_duplicates or target_weights.index.hasnans:
        raise ValueError("target_weights dates must be unique and must not contain NaT")
    if target_weights.columns.has_duplicates:
        raise ValueError("target_weights assets must be unique")
    try:
        validated = target_weights.apply(pd.to_numeric, errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("target_weights must be numeric") from exc
    values = validated.to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("target_weights must contain only finite values")
    if (values < 0.0).any() or (values > 1.0).any():
        raise ValueError("target_weights must be between 0 and 1")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-8, rtol=0.0):
        raise ValueError("target_weights must sum to 1 at every rebalance")
    validated = validated.sort_index()

    figure = Figure(
        figsize=(11, max(4.5, min(12.0, 0.35 * len(validated.columns) + 2.0))),
        layout="constrained",
    )
    axis = figure.subplots()
    image = axis.imshow(
        validated.to_numpy().T, aspect="auto", interpolation="nearest",
        vmin=0.0, vmax=1.0, cmap="Blues",
    )
    ticks = np.linspace(0, len(validated) - 1, min(8, len(validated)), dtype=int)
    axis.set_xticks(ticks, validated.index[ticks].strftime("%Y-%m-%d"))
    axis.tick_params(axis="x", labelrotation=30)
    axis.set_yticks(range(len(validated.columns)), validated.columns)
    axis.set_xlabel("Rebalance date (one column per rebalance)")
    axis.set_ylabel("Asset")
    axis.set_title(title)
    figure.colorbar(image, ax=axis, format=PercentFormatter(xmax=1.0), label="Weight")
    return _save_figure(figure, output_path)
