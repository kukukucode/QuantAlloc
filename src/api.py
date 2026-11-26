"""HTTP interface for the single-strategy QuantAlloc dashboard."""

import logging
import math
import re
from datetime import date, timedelta
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from src.dashboard import run_sample_analysis
from src.dashboard_settings import DashboardSettings
from src.data_provider import fetch_prices
from src.sample_config import (
    ANALYSIS_STRATEGY_LABELS,
    BENCHMARK_TICKER,
    END_DATE,
    START_DATE,
    TICKERS,
)

logger = logging.getLogger(__name__)
app = FastAPI(title="QuantAlloc", version="0.28")
MarketId = Literal["jp", "us"]
MAX_TICKERS = 15
JP_TICKER = re.compile(r"[0-9][0-9A-Z][0-9][0-9A-Z]")
US_TICKER = re.compile(r"[A-Z][A-Z0-9]{0,5}(?:-[A-Z])?")


class MarketConfig(BaseModel):
    label: str
    currency: str
    benchmark_ticker: str
    benchmark_label: str
    default_tickers: list[str]


MARKETS: dict[MarketId, MarketConfig] = {
    "jp": MarketConfig(
        label="日本株",
        currency="JPY",
        benchmark_ticker=BENCHMARK_TICKER,
        benchmark_label="TOPIX",
        default_tickers=TICKERS,
    ),
    "us": MarketConfig(
        label="米国株",
        currency="USD",
        benchmark_ticker="VOO",
        benchmark_label="S&P 500",
        default_tickers=["AAPL", "MSFT", "AMZN", "GOOGL", "JPM"],
    ),
}


class DateRange(BaseModel):
    start_date: date
    end_date: date


class AnalysisRequest(DateRange):
    model_config = ConfigDict(extra="forbid")

    strategy: str
    market: MarketId = "jp"
    tickers: list[str] | None = Field(
        default=None, min_length=1, max_length=MAX_TICKERS
    )


class AnalysisConfig(BaseModel):
    defaults: AnalysisRequest
    strategies: dict[str, str]
    benchmark_ticker: str
    asset_count: int
    markets: dict[MarketId, MarketConfig]
    max_tickers: int
    estimation_window: int
    holding_period: int
    transaction_cost_bps: int
    maximum_date: date


class EvaluationWindow(DateRange):
    observations: int


class PerformanceRow(BaseModel):
    label: str
    cagr: FiniteFloat | None
    volatility: FiniteFloat | None
    sharpe: FiniteFloat | None
    max_drawdown: FiniteFloat | None


class ReturnPoint(BaseModel):
    date: date
    strategy: FiniteFloat
    benchmark: FiniteFloat


class AnalysisResponse(BaseModel):
    requested: AnalysisRequest
    benchmark_label: str
    benchmark_ticker: str
    currency: str
    evaluation: EvaluationWindow
    performance: list[PerformanceRow]
    cumulative_returns: list[ReturnPoint]


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _normalize_tickers(request: AnalysisRequest, market: MarketConfig) -> list[str]:
    if request.tickers is None:
        return list(market.default_tickers)

    tickers = []
    for value in request.tickers:
        ticker = value.strip().upper()
        if request.market == "jp":
            ticker = ticker.removesuffix(".T")
            if not JP_TICKER.fullmatch(ticker):
                raise HTTPException(
                    422, "日本株は4文字の証券コードで入力してください。"
                )
            ticker += ".T"
        else:
            ticker = ticker.replace(".", "-")
            if not US_TICKER.fullmatch(ticker):
                raise HTTPException(
                    422, "米国株はAAPLやBRK-Bの形式で入力してください。"
                )
        tickers.append(ticker)

    if len(set(tickers)) != len(tickers):
        raise HTTPException(422, "重複している銘柄を削除してください。")
    return tickers


@app.get("/api/config")
def configuration() -> AnalysisConfig:
    settings = DashboardSettings()
    return AnalysisConfig(
        defaults=AnalysisRequest(
            strategy="equal_weight",
            start_date=date.fromisoformat(START_DATE),
            end_date=date.fromisoformat(END_DATE) - timedelta(days=1),
            tickers=TICKERS,
        ),
        strategies=ANALYSIS_STRATEGY_LABELS,
        benchmark_ticker=BENCHMARK_TICKER,
        asset_count=len(TICKERS),
        markets=MARKETS,
        max_tickers=MAX_TICKERS,
        estimation_window=settings.estimation_window,
        holding_period=settings.holding_period,
        transaction_cost_bps=round(settings.transaction_cost_rate * 10_000),
        maximum_date=date.today(),
    )


@app.post("/api/analysis")
def analyze(request: AnalysisRequest) -> AnalysisResponse:
    """Analyze the selected market and portfolio over inclusive dates."""
    if request.strategy not in ANALYSIS_STRATEGY_LABELS:
        raise HTTPException(422, "選択できる戦略を指定してください。")
    if not request.start_date <= request.end_date <= date.today():
        raise HTTPException(422, "終了日は開始日以降かつ今日以前にしてください。")

    market = MARKETS[request.market]
    tickers = _normalize_tickers(request, market)
    start = request.start_date.isoformat()
    end_exclusive = (request.end_date + timedelta(days=1)).isoformat()
    try:
        asset_prices = fetch_prices(tickers, start, end_exclusive)
        benchmark_prices = fetch_prices([market.benchmark_ticker], start, end_exclusive)
    except (ValueError, RuntimeError, OSError) as exc:
        logger.warning("Price retrieval failed: %s", exc)
        raise HTTPException(
            502, "株価を取得できませんでした。銘柄コードと取得期間を確認してください。"
        ) from exc

    settings = DashboardSettings()
    minimum_prices = settings.minimum_price_observations
    if len(asset_prices) < minimum_prices:
        raise HTTPException(
            422,
            f"株価は最低{minimum_prices}営業日分必要です。"
            f"取得できたのは{len(asset_prices)}営業日分でした。",
        )

    try:
        result = run_sample_analysis(
            asset_prices,
            benchmark_prices,
            request.strategy,
            settings=settings,
            benchmark_ticker=market.benchmark_ticker,
            benchmark_label=market.benchmark_label,
        )
    except (ValueError, RuntimeError) as exc:
        logger.warning("Analysis failed: %s", exc)
        raise HTTPException(
            422, "指定した条件で分析できませんでした。期間や戦略を変更してください。"
        ) from exc

    summary = result.performance
    return AnalysisResponse(
        requested=request.model_copy(update={"tickers": tickers}),
        benchmark_label=market.benchmark_label,
        benchmark_ticker=market.benchmark_ticker,
        currency=market.currency,
        evaluation=EvaluationWindow(
            start_date=summary.attrs["start_date"].date(),
            end_date=summary.attrs["end_date"].date(),
            observations=summary.attrs["observations"],
        ),
        performance=[
            PerformanceRow(
                label=label,
                cagr=_finite_or_none(row.cagr),
                volatility=_finite_or_none(row.volatility),
                sharpe=_finite_or_none(row.sharpe),
                max_drawdown=_finite_or_none(row.max_drawdown),
            )
            for label, row in summary.iterrows()
        ],
        cumulative_returns=[
            ReturnPoint(date=day.date(), strategy=strategy, benchmark=benchmark)
            for day, strategy, benchmark in result.cumulative_returns.itertuples()
        ],
    )
