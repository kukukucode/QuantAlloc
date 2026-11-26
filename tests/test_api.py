"""Verify the HTTP contract and parity with the Streamlit analysis."""

import asyncio
from pathlib import Path

import httpx
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src import api
from src.sample_config import BENCHMARK_TICKER, TICKERS

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
REQUEST = {
    "strategy": "hierarchical_risk_parity",
    "start_date": "2021-02-01",
    "end_date": "2023-09-01",
}


def call_api(method: str, path: str, **kwargs) -> httpx.Response:
    async def request() -> httpx.Response:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app), base_url="http://test"
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(request())


def test_api_matches_streamlit_for_identical_prices_and_settings(
    sample_market_prices, monkeypatch
) -> None:
    asset_prices, benchmark_prices = sample_market_prices
    calls = []

    def fetch(tickers, start, end):
        calls.append((tickers, start, end))
        prices = benchmark_prices if tickers == [BENCHMARK_TICKER] else asset_prices
        return prices[(prices.index >= start) & (prices.index < end)].copy()

    monkeypatch.setattr(api, "fetch_prices", fetch)
    monkeypatch.setattr("src.data_provider.fetch_prices", fetch)
    response = call_api("POST", "/api/analysis", json=REQUEST)
    assert response.status_code == 200, response.text
    payload = response.json()

    st.cache_data.clear()
    try:
        streamlit = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
        streamlit.selectbox(key="selected_strategy").select(REQUEST["strategy"])
        streamlit.date_input(key="selected_start_date").set_value(
            pd.Timestamp(REQUEST["start_date"]).date()
        )
        streamlit.date_input(key="selected_end_date").set_value(
            pd.Timestamp(REQUEST["end_date"]).date()
        ).run()
        streamlit.button[0].click().run()
        assert not streamlit.exception
        assert not streamlit.error
        result = streamlit.session_state["sample_analysis"]
    finally:
        st.cache_data.clear()

    assert (
        calls
        == [
            (TICKERS, "2021-02-01", "2023-09-02"),
            ([BENCHMARK_TICKER], "2021-02-01", "2023-09-02"),
        ]
        * 2
    )
    assert payload["requested"] == REQUEST | {"market": "jp", "tickers": TICKERS}
    assert payload["benchmark_label"] == "TOPIX"
    assert payload["currency"] == "JPY"
    summary = result.performance
    assert payload["evaluation"] == {
        "start_date": summary.attrs["start_date"].date().isoformat(),
        "end_date": summary.attrs["end_date"].date().isoformat(),
        "observations": summary.attrs["observations"],
    }
    assert [row["label"] for row in payload["performance"]] == list(summary.index)
    for row in payload["performance"]:
        expected = summary.loc[row["label"]]
        for metric in ("cagr", "volatility", "sharpe", "max_drawdown"):
            assert row[metric] == pytest.approx(expected[metric], abs=1e-12)

    points = pd.DataFrame(payload["cumulative_returns"])
    assert points["date"].tolist() == [
        day.date().isoformat() for day in result.cumulative_returns.index
    ]
    assert points["strategy"].tolist() == pytest.approx(
        result.cumulative_returns.iloc[:, 0].tolist(), abs=1e-12
    )
    assert points["benchmark"].tolist() == pytest.approx(
        result.cumulative_returns["TOPIX"].tolist(), abs=1e-12
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"start_date": "2024-01-01"},
        {"end_date": "invalid"},
        {"end_date": "9999-12-31"},
        {"strategy": "unknown"},
        {"market": "unknown"},
        {"market": "us", "tickers": ["7203.T"]},
        {"market": "us", "tickers": ["BTC-USD"]},
        {"market": "jp", "tickers": ["AAPL"]},
        {"tickers": ["7203", "7203.t"]},
        {"tickers": []},
        {"tickers": [str(7200 + index) for index in range(api.MAX_TICKERS + 1)]},
    ],
)
def test_api_rejects_invalid_requests_before_fetching(monkeypatch, overrides) -> None:
    def unexpected_fetch(*args):
        pytest.fail("invalid requests must not fetch prices")

    monkeypatch.setattr(api, "fetch_prices", unexpected_fetch)
    response = call_api("POST", "/api/analysis", json=REQUEST | overrides)
    assert response.status_code == 422


def test_api_reports_insufficient_prices(monkeypatch, sample_market_prices) -> None:
    asset_prices, benchmark_prices = sample_market_prices

    def fetch(tickers, *args):
        return (
            benchmark_prices
            if tickers == [BENCHMARK_TICKER]
            else asset_prices.iloc[:10]
        )

    monkeypatch.setattr(api, "fetch_prices", fetch)
    response = call_api("POST", "/api/analysis", json=REQUEST)
    assert response.status_code == 422
    assert "568営業日分" in response.json()["detail"]
    assert "10営業日分" in response.json()["detail"]


@pytest.mark.parametrize(
    ("market", "inputs", "tickers", "benchmark", "label", "currency"),
    [
        ("jp", [" 130a ", "6758.t"], ["130A.T", "6758.T"], "1306.T", "TOPIX", "JPY"),
        ("us", ["brk.b", "msft"], ["BRK-B", "MSFT"], "VOO", "S&P 500", "USD"),
    ],
)
def test_api_uses_selected_market_and_tickers(
    monkeypatch,
    sample_market_prices,
    market,
    inputs,
    tickers,
    benchmark,
    label,
    currency,
) -> None:
    asset_prices, benchmark_prices = sample_market_prices
    calls = []

    def fetch(symbols, *args):
        calls.append(symbols)
        if symbols == [benchmark]:
            return benchmark_prices.set_axis(symbols, axis="columns")
        return asset_prices.iloc[:, : len(symbols)].set_axis(symbols, axis="columns")

    monkeypatch.setattr(api, "fetch_prices", fetch)
    response = call_api(
        "POST", "/api/analysis", json=REQUEST | {"market": market, "tickers": inputs}
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert calls == [tickers, [benchmark]]
    assert payload["requested"]["tickers"] == tickers
    assert payload["benchmark_ticker"] == benchmark
    assert payload["benchmark_label"] == label
    assert payload["currency"] == currency
    assert payload["performance"][1]["label"] == label
    assert len(payload["cumulative_returns"]) == payload["evaluation"]["observations"]


def test_api_serializes_undefined_sharpe_as_null(
    monkeypatch, sample_market_prices
) -> None:
    def fetch(tickers, *args):
        return pd.DataFrame(100.0, index=sample_market_prices[0].index, columns=tickers)

    monkeypatch.setattr(api, "fetch_prices", fetch)
    response = call_api(
        "POST", "/api/analysis", json=REQUEST | {"strategy": "equal_weight"}
    )
    assert response.status_code == 200, response.text
    assert all(row["sharpe"] is None for row in response.json()["performance"])
