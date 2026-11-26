"""Verify the public analysis API and its coverage rules."""

import asyncio

import httpx
import pandas as pd
import pytest

from src import api
from src.sample_config import BENCHMARK_TICKER, TICKERS

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


def test_api_uses_prior_training_for_short_evaluation(
    sample_market_prices, monkeypatch
):
    assets, benchmark = sample_market_prices
    calls = []

    def fetch(tickers, start, end, *, fill_missing):
        calls.append((start, end, fill_missing))
        prices = benchmark if tickers == [BENCHMARK_TICKER] else assets
        return prices.loc[(prices.index >= start) & (prices.index < end)]

    monkeypatch.setattr(api, "fetch_prices", fetch)
    response = call_api(
        "POST",
        "/api/analysis",
        json=REQUEST | {"start_date": "2023-04-17", "end_date": "2023-04-21"},
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert all(start < "2021-01-01" and end == "2023-04-22" for start, end, _ in calls)
    assert all(not filled for _, _, filled in calls)
    assert result["applied_strategy"] == REQUEST["strategy"]
    assert result["evaluation"] == {
        "start_date": "2023-04-17",
        "end_date": "2023-04-21",
        "observations": 5,
    }
    assert result["requested"]["tickers"] == TICKERS
    assert [row["label"] for row in result["performance"]] == [
        "Hierarchical Risk Parity (HRP)",
        "TOPIX",
    ]
    assert not result["annualized_metrics_available"]
    assert all(
        row["cagr"] is None and row["period_return"] is not None
        for row in result["performance"]
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


def test_equal_weight_adds_new_listing_after_its_first_close(monkeypatch):
    dates = pd.bdate_range("2024-09-02", periods=5)
    assets = pd.DataFrame(100.0, index=dates, columns=TICKERS)
    new_ticker = TICKERS[-1]
    assets.loc[dates[:2], new_ticker] = float("nan")
    assets.loc[dates[2] :, new_ticker] = [200.0, 220.0, 220.0]
    benchmark = pd.DataFrame(100.0, index=dates, columns=[BENCHMARK_TICKER])

    def fetch(tickers, *args, **kwargs):
        return benchmark if tickers == [BENCHMARK_TICKER] else assets

    monkeypatch.setattr(api, "fetch_prices", fetch)
    params = REQUEST | {
        "strategy": "equal_weight",
        "start_date": "2024-09-03",
        "end_date": "2024-09-06",
    }
    response = call_api("POST", "/api/analysis", json=params)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["evaluation"]["observations"] == 4
    assert [result["entry_dates"][ticker] for ticker in TICKERS[:-1]] == [
        "2024-09-03"
    ] * 9
    assert result["entry_dates"][new_ticker] == "2024-09-05"
    assert result["unavailable_tickers"] == []
    assert [
        point["strategy"] for point in result["cumulative_returns"][:2]
    ] == pytest.approx([0.0, 0.0])
    # The new stock's first +10% day contributes 1%, less the cost of buying 10%.
    assert result["cumulative_returns"][2]["strategy"] == pytest.approx(
        1.01 * (1 - 0.2 * 0.001) - 1
    )
    assert (
        result["performance"][0]["period_return"]
        == result["cumulative_returns"][-1]["strategy"]
    )
    assert all(
        row["sharpe"] is None and row["volatility"] is None
        for row in result["performance"]
    )

    fallback = call_api(
        "POST", "/api/analysis", json=params | {"strategy": "minimum_variance"}
    )
    assert fallback.status_code == 200
    assert fallback.json()["applied_strategy"] == "equal_weight"


def test_api_can_show_one_return_before_a_later_listing(monkeypatch):
    dates = pd.bdate_range("2024-09-02", periods=4)
    tickers = ["7203.T", "130A.T"]
    prices = pd.DataFrame(
        {
            tickers[0]: [100.0, 102.0, 103.0, 104.0],
            tickers[1]: [None, None, 100.0, 120.0],
        },
        index=dates,
    )

    def fetch(symbols, *args, **kwargs):
        if symbols == [BENCHMARK_TICKER]:
            return pd.DataFrame(100.0, index=dates, columns=symbols)
        return prices[symbols]

    monkeypatch.setattr(api, "fetch_prices", fetch)
    response = call_api(
        "POST",
        "/api/analysis",
        json=REQUEST
        | {
            "strategy": "equal_weight",
            "tickers": tickers,
            "start_date": "2024-09-03",
            "end_date": "2024-09-03",
        },
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["evaluation"]["observations"] == 1
    assert result["unavailable_tickers"] == [tickers[1]]
    assert result["performance"][0]["period_return"] == pytest.approx(0.02)
    assert result["performance"][0]["cagr"] is None


def test_api_does_not_count_returns_from_before_a_price_gap(monkeypatch):
    dates = pd.bdate_range("2024-09-02", periods=5)
    prices = pd.DataFrame({"7203.T": [100.0, None, 200.0, 210.0, None]}, index=dates)

    def fetch(symbols, *args, **kwargs):
        if symbols == [BENCHMARK_TICKER]:
            return pd.DataFrame(100.0, index=dates, columns=symbols)
        return prices

    monkeypatch.setattr(api, "fetch_prices", fetch)
    response = call_api(
        "POST",
        "/api/analysis",
        json=REQUEST
        | {
            "strategy": "equal_weight",
            "tickers": ["7203"],
            "start_date": "2024-09-04",
            "end_date": "2024-09-06",
        },
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["evaluation"] == {
        "start_date": "2024-09-05",
        "end_date": "2024-09-05",
        "observations": 1,
    }
    assert result["performance"][0]["period_return"] == pytest.approx(0.05)


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

    def fetch(symbols, *args, **kwargs):
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
    def fetch(tickers, *args, **kwargs):
        return pd.DataFrame(100.0, index=sample_market_prices[0].index, columns=tickers)

    monkeypatch.setattr(api, "fetch_prices", fetch)
    response = call_api(
        "POST", "/api/analysis", json=REQUEST | {"strategy": "equal_weight"}
    )
    assert response.status_code == 200, response.text
    assert all(row["sharpe"] is None for row in response.json()["performance"])
