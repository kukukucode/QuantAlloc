"""Offline integration checks for the sample analysis workflow."""

import numpy as np
import pandas as pd

import main as application


def test_main_reuses_backtests_across_comparisons(monkeypatch, capsys) -> None:
    assets = ["A", "B", "C"]
    symbols = assets + [application.BENCHMARK_TICKER]
    dates = pd.bdate_range("2021-01-01", periods=757)
    generator = np.random.default_rng(19)
    returns = generator.normal(0.0005, 0.01, size=(757, 4))
    prices = pd.DataFrame(
        100.0 * np.cumprod(1.0 + returns, axis=0), index=dates, columns=symbols
    )
    monkeypatch.setattr(application, "WEIGHTS", dict.fromkeys(assets, 1.0 / 3.0))
    monkeypatch.setattr(application, "TICKERS", assets)
    monkeypatch.setattr(
        application, "fetch_prices", lambda tickers, start, end: prices[tickers].copy()
    )
    original_backtest = application.walk_forward_backtest
    calls = []

    def recording_backtest(asset_returns, strategy, **kwargs):
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

    assert len(calls) == len(set(calls)) == 18
    output = capsys.readouterr().out
    assert "Realistic OOS Performance" in output
    assert "Rebalancing Frequency Comparison" in output
    assert "Covariance Estimation OOS Comparison" in output
