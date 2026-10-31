"""Exercise the real Streamlit script without network access."""

from pathlib import Path

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.dashboard import run_sample_analysis
from src.sample_config import (
    BENCHMARK_TICKER,
    END_DATE,
    START_DATE,
    STRATEGY_LABELS,
    TICKERS,
)

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture(autouse=True)
def clear_app_cache():
    st.cache_data.clear()
    yield
    st.cache_data.clear()


@pytest.fixture
def offline_fetch(monkeypatch, sample_market_prices):
    asset_prices, benchmark_prices = sample_market_prices
    calls = []

    def fetch(tickers, start, end):
        calls.append((list(tickers), start, end))
        return (
            benchmark_prices.copy()
            if tickers == [BENCHMARK_TICKER]
            else asset_prices.loc[:, tickers].copy()
        )

    monkeypatch.setattr("src.data_provider.fetch_prices", fetch)
    return calls


def test_app_waits_for_explicit_run(offline_fetch) -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value == "QuantAlloc"
    assert app.button[0].label == "分析を実行"
    assert not app.metric
    assert offline_fetch == []
    assert app.info
    assert app.selectbox[0].options == list(STRATEGY_LABELS.values())


@pytest.mark.parametrize("strategy", list(STRATEGY_LABELS))
def test_app_displays_net_metrics_and_percentage_charts(
    offline_fetch, sample_market_prices, monkeypatch, strategy
) -> None:
    charts = []
    original_line_chart = st.line_chart

    def recording_chart(data, **kwargs):
        charts.append(data.copy())
        return original_line_chart(data, **kwargs)

    monkeypatch.setattr(st, "line_chart", recording_chart)
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    app.selectbox[0].select(strategy).run()
    assert offline_fetch == []
    app.button[0].click().run()
    assert not app.exception
    assert not app.error
    assert len(app.metric) == 4
    label = STRATEGY_LABELS[strategy]
    expected = run_sample_analysis(*sample_market_prices, strategy)
    assert app.metric[0].value == f"{expected.performance.loc[label, 'cagr']:.2%}"
    table = next(
        frame.value for frame in app.dataframe if "CAGR" in frame.value.columns
    )
    assert table.index.to_list() == [label, "TOPIX"]
    assert any(heading.value == f"{label} vs TOPIX" for heading in app.subheader)
    assert table.loc["TOPIX", "CAGR"] == pytest.approx(
        expected.performance.loc["TOPIX", "cagr"]
    )
    assert len(charts) == 2
    pd.testing.assert_frame_equal(charts[0], expected.cumulative_returns * 100.0)
    pd.testing.assert_frame_equal(charts[1], expected.drawdowns * 100.0)
    assert offline_fetch == [
        (TICKERS, START_DATE, END_DATE),
        ([BENCHMARK_TICKER], START_DATE, END_DATE),
    ]
    assert any("売買コストは推定しません" in caption.value for caption in app.caption)


def test_app_preserves_result_on_rerun_and_reuses_price_cache(
    offline_fetch, monkeypatch
) -> None:
    original_analysis = run_sample_analysis
    calls = []

    def recording_analysis(*args):
        calls.append(1)
        return original_analysis(*args)

    monkeypatch.setattr("src.dashboard.run_sample_analysis", recording_analysis)
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    app.button[0].click().run()
    app.run()
    assert not app.exception
    assert len(app.metric) == 4
    assert len(calls) == 1
    assert len(offline_fetch) == 2
    app.button[0].click().run()
    assert not app.exception
    assert len(calls) == 2
    assert len(offline_fetch) == 2


def test_app_shows_download_error_without_crashing(monkeypatch) -> None:
    def fail_fetch(*args):
        raise RuntimeError("download unavailable")

    monkeypatch.setattr("src.data_provider.fetch_prices", fail_fetch)
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    app.button[0].click().run()
    assert not app.exception
    assert "download unavailable" in app.error[0].value
    assert not app.metric


def test_app_discards_stale_result_after_failed_refresh(
    offline_fetch, monkeypatch
) -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    app.button[0].click().run()
    assert len(app.metric) == 4
    st.cache_data.clear()

    def fail_fetch(*args):
        raise ValueError("no complete price observations were returned")

    monkeypatch.setattr("src.data_provider.fetch_prices", fail_fetch)
    app.button[0].click().run()
    assert not app.exception
    assert app.error
    assert not app.metric
    assert "sample_analysis" not in app.session_state


def test_app_reports_insufficient_history(
    offline_fetch, sample_market_prices, monkeypatch
) -> None:
    asset_prices, benchmark_prices = sample_market_prices

    def short_fetch(tickers, start, end):
        return (
            benchmark_prices
            if tickers == [BENCHMARK_TICKER]
            else asset_prices.iloc[:550]
        )

    monkeypatch.setattr("src.data_provider.fetch_prices", short_fetch)
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    app.button[0].click().run()
    assert not app.exception
    assert "567 observations" in app.error[0].value
    assert not app.metric


def test_app_invalidates_result_and_cache_after_date_change(
    offline_fetch, monkeypatch
) -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    app.button[0].click().run()
    assert len(app.metric) == 4
    monkeypatch.setattr("src.sample_config.START_DATE", "2021-04-01")
    app.run()
    assert not app.metric
    app.button[0].click().run()
    assert not app.exception
    assert len(offline_fetch) == 4
    assert [call[1] for call in offline_fetch[-2:]] == ["2021-04-01"] * 2


@pytest.mark.parametrize("changed_argument", [0, 1, 2, 3])
def test_price_cache_key_includes_every_request_argument(
    monkeypatch, changed_argument
) -> None:
    import app as application

    calls = []

    def fetch(tickers, start, end):
        calls.append((tuple(tickers), start, end))
        return pd.DataFrame(
            100.0, index=pd.bdate_range(start, periods=2), columns=tickers
        )

    monkeypatch.setattr(application, "fetch_prices", fetch)
    request = [("A", "B"), "TOPIX", "2021-01-01", "2026-01-01"]
    application.load_sample_prices(*request)
    application.load_sample_prices(*request)
    assert len(calls) == 2
    request[changed_argument] = [("A", "C"), "OTHER", "2022-01-01", "2025-01-01"][
        changed_argument
    ]
    application.load_sample_prices(*request)
    assert len(calls) == 4


def test_app_displays_undefined_sharpe_as_unavailable(
    monkeypatch, sample_market_prices
) -> None:
    dates = sample_market_prices[0].index

    def flat_fetch(tickers, start, end):
        return pd.DataFrame(100.0, index=dates, columns=tickers)

    monkeypatch.setattr("src.data_provider.fetch_prices", flat_fetch)
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    app.button[0].click().run()
    assert not app.exception
    sharpe = next(metric for metric in app.metric if metric.label == "Sharpe")
    assert sharpe.value == "算出不可"


def test_strategy_change_clears_old_result_without_refetching(
    offline_fetch, monkeypatch
) -> None:
    calls = []

    def recording_analysis(asset_prices, benchmark_prices, strategy):
        calls.append(strategy)
        return run_sample_analysis(asset_prices, benchmark_prices, strategy)

    monkeypatch.setattr("src.dashboard.run_sample_analysis", recording_analysis)
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    app.button[0].click().run()
    assert len(app.metric) == 4
    app.selectbox[0].select("minimum_variance").run()
    assert not app.metric
    assert calls == ["equal_weight"]
    assert len(offline_fetch) == 2
    app.button[0].click().run()
    assert not app.exception
    assert len(app.metric) == 4
    assert calls == ["equal_weight", "minimum_variance"]
    assert len(offline_fetch) == 2
    assert app.session_state["sample_analysis"].strategy == "minimum_variance"


def test_failed_strategy_is_reported_without_fallback(
    offline_fetch, monkeypatch
) -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    app.button[0].click().run()
    assert len(app.metric) == 4
    attempts = []

    def fail_analysis(asset_prices, benchmark_prices, strategy):
        attempts.append(strategy)
        raise RuntimeError("portfolio optimization failed")

    monkeypatch.setattr("src.dashboard.run_sample_analysis", fail_analysis)
    app.selectbox[0].select("minimum_variance").run()
    app.button[0].click().run()
    assert not app.exception
    assert "portfolio optimization failed" in app.error[0].value
    assert attempts == ["minimum_variance"]
    assert not app.metric
    assert "sample_analysis" not in app.session_state


def test_app_displays_tail_risk_table(offline_fetch, sample_market_prices) -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()

    app.button[0].click().run()
    assert not app.exception

    tables = [
        item.value
        for item in app.dataframe
        if {"VaR 95%", "CVaR 95%"}.issubset(item.value.columns)
    ]
    assert len(tables) == 1

    displayed = tables[0]
    expected = run_sample_analysis(*sample_market_prices).tail_risk

    assert displayed.index.tolist() == ["Equal Weight", "TOPIX"]

    for name in expected.index:
        assert displayed.loc[name, "VaR 95%"] == pytest.approx(
            expected.loc[name, "var_95"]
        )
        assert displayed.loc[name, "CVaR 95%"] == pytest.approx(
            expected.loc[name, "cvar_95"]
        )


def test_app_compares_all_strategies_and_reuses_cache(
    offline_fetch,
) -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()

    assert len(app.button) == 3
    assert app.button[1].label == "4戦略を比較"

    app.button[1].click().run()
    assert not app.exception
    assert not app.error
    assert not app.metric

    result = app.session_state["sample_comparison"]
    expected_labels = list(STRATEGY_LABELS.values()) + ["TOPIX"]

    assert result.performance.index.tolist() == expected_labels
    assert result.cumulative_returns.columns.tolist() == expected_labels
    assert result.drawdowns.columns.tolist() == expected_labels

    performance_table = next(
        item.value for item in app.dataframe if "CAGR" in item.value.columns
    )
    risk_table = next(
        item.value for item in app.dataframe if "CVaR 95%" in item.value.columns
    )

    assert performance_table.index.tolist() == expected_labels
    assert risk_table.index.tolist() == expected_labels
    assert len(offline_fetch) == 2

    app.button[0].click().run()
    assert not app.exception
    assert "sample_comparison" not in app.session_state
    assert len(app.metric) == 4
    assert len(offline_fetch) == 2


def test_app_displays_rolling_risk_comparison(offline_fetch, monkeypatch) -> None:
    charts = []
    original_line_chart = st.line_chart

    def recording_chart(data, **kwargs):
        charts.append(data.copy())
        return original_line_chart(data, **kwargs)

    monkeypatch.setattr(st, "line_chart", recording_chart)

    app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()

    app.button[1].click().run()

    assert not app.exception
    assert not app.error
    assert len(app.selectbox) == 2
    rolling_selector = next(
        box for box in app.selectbox if box.label == "ローリング期間"
    )
    assert rolling_selector.value == 63

    result = app.session_state["sample_comparison"]

    assert len(charts) == 5
    pd.testing.assert_frame_equal(
        charts[2],
        result.rolling_volatility[63].dropna(how="all") * 100.0,
    )
    pd.testing.assert_frame_equal(
        charts[3],
        result.rolling_sharpe[63].dropna(how="all"),
    )

    assert len(offline_fetch) == 2


def test_app_displays_stress_comparison(offline_fetch) -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()

    app.button[1].click().run()

    assert not app.exception
    assert not app.error

    result = app.session_state["sample_comparison"]

    table = next(
        item.value for item in app.dataframe if "Stress Return" in item.value.columns
    )

    assert table.index.tolist() == (list(STRATEGY_LABELS.values()) + ["TOPIX"])

    assert table.loc["TOPIX", "Stress Return"] == pytest.approx(
        result.stress.summary.loc["TOPIX", "window_return"]
    )

    assert len(offline_fetch) == 2


def test_app_displays_factor_exposure(offline_fetch) -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()

    app.button[1].click().run()

    assert not app.exception
    assert not app.error

    result = app.session_state["sample_comparison"]

    table = next(
        item.value for item in app.dataframe if "Market Beta" in item.value.columns
    )

    expected_labels = list(STRATEGY_LABELS.values()) + ["TOPIX"]

    assert table.index.tolist() == expected_labels

    for name in expected_labels:
        assert table.loc[name, "Market Beta"] == pytest.approx(
            result.factor_exposure.loc[name, "beta"]
        )

        assert table.loc[name, "Annual Alpha"] == pytest.approx(
            result.factor_exposure.loc[name, "alpha_annual"]
        )

        assert table.loc[name, "R²"] == pytest.approx(
            result.factor_exposure.loc[name, "r_squared"]
        )

    assert len(offline_fetch) == 2


def test_app_runs_sensitivity_only_on_request(
    offline_fetch,
) -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=90).run()

    assert len(app.button) == 3
    assert app.button[2].label == "感応度を分析"
    assert offline_fetch == []

    app.button[2].click().run()

    assert not app.exception
    assert not app.error
    assert not app.metric

    result = app.session_state["sample_sensitivity"]

    assert result.strategy == "equal_weight"
    assert len(offline_fetch) == 2

    tables = [
        item.value for item in app.dataframe if "Total Cost" in item.value.columns
    ]

    assert len(tables) == 3
    assert "Insufficient history" in tables[1]["Status"].values

    app.button[0].click().run()

    assert not app.exception
    assert "sample_sensitivity" not in app.session_state
    assert len(app.metric) == 4
    assert len(offline_fetch) == 2
