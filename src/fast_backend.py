"""Optional Rust holding simulation with an exact Python compatibility path.

QUANTALLOC_ENGINE=python pins the original pandas path.
QUANTALLOC_ENGINE=rust requires the compiled extension (fail closed).
QUANTALLOC_ENGINE=auto uses Rust for larger inputs when installed.
"""

import importlib
import os

import pandas as pd

# Heuristic only. benchmarks/holding_kernel.py provides measured comparisons.
_AUTO_RUST_MIN_ELEMENTS = 4096


def _python_holding(
    target_weights: pd.Series,
    holding_returns: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Original pandas reference, retained for compatibility and replay."""
    asset_values = (1.0 + holding_returns).cumprod().mul(
        target_weights,
        axis="columns",
    )
    portfolio_values = asset_values.sum(axis=1)
    if (portfolio_values <= 0.0).any():
        raise ValueError("portfolio value must remain positive")

    gross_returns = portfolio_values.pct_change(fill_method=None)
    gross_returns.iloc[0] = portfolio_values.iloc[0] - 1.0
    gross_returns.name = "gross_return"

    ending_weights = asset_values.iloc[-1] / portfolio_values.iloc[-1]
    ending_weights.name = "pre_rebalance_weight"
    return gross_returns, ending_weights


def simulate_holding_period(
    target_weights: pd.Series,
    holding_returns: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Preserve output names, indexes, drift and transaction-cost semantics."""
    mode = os.environ.get("QUANTALLOC_ENGINE", "auto").lower()
    if mode not in {"auto", "python", "rust"}:
        raise ValueError("QUANTALLOC_ENGINE must be auto, python, or rust")

    size = holding_returns.shape[0] * holding_returns.shape[1]
    if mode == "python" or (
        mode == "auto" and size < _AUTO_RUST_MIN_ELEMENTS
    ):
        return _python_holding(target_weights, holding_returns)

    try:
        native = importlib.import_module("_quantalloc_rust")
    except ImportError as exc:
        if mode == "rust":
            raise RuntimeError(
                "QUANTALLOC_ENGINE=rust requires the quantalloc-fast extension"
            ) from exc
        return _python_holding(target_weights, holding_returns)

    ordered_weights = target_weights.reindex(holding_returns.columns)
    gross, ending = native.simulate_holding(
        holding_returns.to_numpy(dtype=float, copy=False),
        ordered_weights.to_numpy(dtype=float, copy=False),
    )
    return (
        pd.Series(gross, index=holding_returns.index, name="gross_return"),
        pd.Series(
            ending,
            index=holding_returns.columns,
            name="pre_rebalance_weight",
        ),
    )
