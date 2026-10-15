"""Exercise the real Streamlit script without network access."""

from pathlib import Path

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.dashboard import run_sample_analysis
from src.sample_config import BENCHMARK_TICKER, END_DATE, START_DATE, TICKERS


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


def test_app_displays_net_metrics_and_percentage_charts(
    offline_fetch, sample_market_prices, monkeypatch
) -> None:
    charts = []
    original_line_chart = st.line_chart

    def recording_chart(data, **kwargs):
        charts.append(data.copy())
        return original_line_chart(data, **kwargs)

    monkeypatch.setattr(st, "line_chart", recording_chart)
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    app.button[0].click().run()
    assert not app.exception
    assert not app.error
    assert len(app.metric) == 4
    expected = run_sample_analysis(*sample_market_prices)
    assert app.metric[0].value == f"{expected.performance.loc['Equal Weight', 'cagr']:.2%}"
    table = next(frame.value for frame in app.dataframe if "CAGR" in frame.value.columns)
    assert table.index.to_list() == ["Equal Weight", "TOPIX"]
    assert table.loc["TOPIX", "CAGR"] == pytest.approx(expected.performance.loc["TOPIX", "cagr"])
    assert len(charts) == 2
    pd.testing.assert_frame_equal(charts[0], expected.cumulative_returns * 100.0)
    pd.testing.assert_frame_equal(charts[1], expected.drawdowns * 100.0)
    assert offline_fetch == [
        (TICKERS, START_DATE, END_DATE), ([BENCHMARK_TICKER], START_DATE, END_DATE)
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


def test_app_discards_stale_result_after_failed_refresh(offline_fetch, monkeypatch) -> None:
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


def test_app_reports_insufficient_history(offline_fetch, sample_market_prices, monkeypatch) -> None:
    asset_prices, benchmark_prices = sample_market_prices

    def short_fetch(tickers, start, end):
        return benchmark_prices if tickers == [BENCHMARK_TICKER] else asset_prices.iloc[:550]

    monkeypatch.setattr("src.data_provider.fetch_prices", short_fetch)
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    app.button[0].click().run()
    assert not app.exception
    assert "567 observations" in app.error[0].value
    assert not app.metric
