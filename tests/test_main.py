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


@pytest.mark.parametrize("start_position", [4, 5, 6])
def test_oos_turnover_and_costs_cover_only_comparison_period(
    monkeypatch, capsys, start_position: int
) -> None:
    dates = pd.bdate_range("2024-01-01", periods=10)
    returns = pd.DataFrame(
        {
            "A": [0.0, 0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0, 0.0, 0.0],
            "B": [0.0] * 10,
        },
        index=dates,
    )
    result = application.walk_forward_backtest(
        returns, "equal_weight", estimation_window=2, holding_period=2,
        transaction_cost_rate=0.01,
    )
    benchmark = pd.Series(
        [0.01, -0.01], index=dates[start_position:start_position + 2]
    )
    gross = application.compare_strategies(
        {"Equal Weight": result.gross_returns, "TOPIX": benchmark}
    )
    net = application.compare_strategies(
        {"Equal Weight": result.net_returns, "TOPIX": benchmark}
    )
    results = {"equal_weight": result}
    turnover = application._compare_oos_turnover(results, net)
    start, end = benchmark.index[0], benchmark.index[-1]
    common_turnover = result.turnover.loc[start:end]
    expected_average = (
        float(common_turnover.mean()) if not common_turnover.empty else 0.0
    )
    assert turnover.loc["Equal Weight", "average_turnover"] == pytest.approx(
        expected_average
    )
    assert turnover.loc["Equal Weight", "total_turnover"] == pytest.approx(
        common_turnover.sum()
    )
    expected_maximum = (
        float(common_turnover.max()) if not common_turnover.empty else 0.0
    )
    assert turnover.loc["Equal Weight", "maximum_turnover"] == pytest.approx(
        expected_maximum
    )
    expected_cost = result.transaction_costs.loc[start:end].sum()
    assert expected_cost < result.transaction_costs.sum()

    monkeypatch.setattr(
        application, "STRATEGY_LABELS", {"equal_weight": "Equal Weight"}
    )
    application.print_strategy_comparison(net, turnover)
    strategy_row = next(
        line for line in capsys.readouterr().out.splitlines()
        if line.startswith("Equal Weight")
    )
    assert strategy_row.split()[-1] == f"{expected_average:.2%}"
    application.print_gross_net_comparison(gross, net, results)
    cost_row = next(
        line for line in capsys.readouterr().out.splitlines()
        if line.startswith("Equal Weight")
    )
    assert cost_row.split()[-1] == f"{expected_cost:.2%}"


def test_oos_turnover_is_zero_when_common_period_has_no_rebalance() -> None:
    dates = pd.bdate_range("2024-01-01", periods=8)
    returns = pd.DataFrame({"A": [0.0] * 8, "B": [0.0] * 8}, index=dates)
    result = application.walk_forward_backtest(
        returns, "equal_weight", estimation_window=2, holding_period=3
    )
    comparison = application.compare_strategies(
        {
            "Equal Weight": result.net_returns,
            "TOPIX": pd.Series([0.01, -0.01], index=dates[3:5]),
        }
    )
    turnover = application._compare_oos_turnover({"equal_weight": result}, comparison)
    assert turnover.loc["Equal Weight"].to_list() == [0.0, 0.0, 0.0]
