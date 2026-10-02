"""Offline integration checks for the sample analysis workflow."""

import numpy as np
import pandas as pd
import pytest

import main as application


@pytest.mark.parametrize("use_sample_portfolio", [False, True])
def test_main_reuses_backtests_across_comparisons(
    monkeypatch, capsys, use_sample_portfolio: bool
) -> None:
    assets = application.TICKERS if use_sample_portfolio else ["A", "B", "C"]
    symbols = assets + [application.BENCHMARK_TICKER]
    dates = pd.bdate_range("2021-01-01", periods=757)
    generator = np.random.default_rng(19)
    returns = generator.normal(0.0005, 0.01, size=(757, len(symbols)))
    prices = pd.DataFrame(
        100.0 * np.cumprod(1.0 + returns, axis=0), index=dates, columns=symbols
    )
    monkeypatch.setattr(application, "WEIGHTS", dict.fromkeys(assets, 1.0 / len(assets)))
    monkeypatch.setattr(application, "TICKERS", assets)
    monkeypatch.setattr(
        application, "fetch_prices", lambda tickers, start, end: prices[tickers].copy()
    )
    original_backtest = application.walk_forward_backtest
    calls = []
    bl_configs = []

    def recording_backtest(asset_returns, strategy, **kwargs):
        if strategy == "black_litterman":
            bl_configs.append(kwargs["black_litterman"])
        calls.append(
            (
                strategy,
                kwargs.get("holding_period", 63),
                kwargs.get("covariance_method", "sample"),
            )
        )
        return original_backtest(asset_returns, strategy, **kwargs)

    monkeypatch.setattr(application, "walk_forward_backtest", recording_backtest)

    application.main()

    assert len(calls) == len(set(calls)) == 19
    assert len(bl_configs) == 1
    config = bl_configs[0]
    assert config.market_weights.to_list() == pytest.approx([1.0 / len(assets)] * len(assets))
    assert config.view_matrix.iloc[0].to_list() == [1.0, -1.0] + [0.0] * (len(assets) - 2)
    output = capsys.readouterr().out
    assert "Realistic OOS Performance" in output
    assert "Rebalancing Frequency Comparison" in output
    assert "Covariance Estimation OOS Comparison" in output
    assert "Black-Litterman Assumptions (illustrative)" in output
    assert f"View: {assets[0]} - {assets[1]} = 2.00%" in output
    assert "Historical Mean vs Black-Litterman OOS" in output
    assert "Historical Mean" in output
    assert "Sum Cost" in output
    assert "Expected Return OOS Period" in output
