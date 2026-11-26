"""HTTP interface for the single-strategy QuantAlloc dashboard."""

import logging
import math
from datetime import date, timedelta
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from src.dashboard_settings import DashboardSettings
from src.data_provider import fetch_prices
from src.metrics import (
    annualized_volatility,
    cagr,
    max_drawdown,
    sharpe_ratio,
    wealth_index,
)
from src.sample_config import (
    ANALYSIS_STRATEGY_LABELS,
    BENCHMARK_TICKER,
    END_DATE,
    START_DATE,
    TICKERS,
)
from src.symbol_search import (
    JP_TICKER,
    US_TICKER,
    SymbolMatch,
    normalize_symbol_query,
    search_japanese_symbols,
    search_us_symbols,
)
from src.web_analysis import NoEvaluationData, run_web_analysis

logger = logging.getLogger(__name__)
app = FastAPI(title="QuantAlloc", version="0.30")
MarketId = Literal["jp", "us"]
MAX_TICKERS = 15
MIN_ANNUALIZED_OBSERVATIONS = 126


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
    period_return: FiniteFloat
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
    applied_strategy: str
    benchmark_label: str
    benchmark_ticker: str
    currency: str
    evaluation: EvaluationWindow
    annualized_metrics_available: bool
    minimum_annualized_observations: int
    entry_dates: dict[str, date]
    unavailable_tickers: list[str]
    skipped_sessions: int
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


@app.get("/api/symbols")
def search_symbols(
    market: MarketId, q: Annotated[str, Query(min_length=2, max_length=80)]
) -> list[SymbolMatch]:
    query = normalize_symbol_query(q)
    if len(query) < 2:
        raise HTTPException(422, "検索語は2文字以上入力してください。")
    try:
        if market == "jp":
            return search_japanese_symbols(query)
        return search_us_symbols(query)
    except Exception as exc:
        logger.warning("Symbol search failed for market %s: %s", market, exc)
        raise HTTPException(
            502, "銘柄検索に接続できませんでした。時間をおいて再度お試しください。"
        ) from exc


@app.post("/api/analysis")
def analyze(request: AnalysisRequest) -> AnalysisResponse:
    """Analyze the selected market and portfolio over inclusive dates."""
    if request.strategy not in ANALYSIS_STRATEGY_LABELS:
        raise HTTPException(422, "選択できる戦略を指定してください。")
    if not request.start_date <= request.end_date <= date.today():
        raise HTTPException(422, "終了日は開始日以降かつ今日以前にしてください。")

    market = MARKETS[request.market]
    tickers = _normalize_tickers(request, market)
    lookback_days = 31 if request.strategy == "equal_weight" else 1095
    start = date.fromordinal(
        max(1, request.start_date.toordinal() - lookback_days)
    ).isoformat()
    end_exclusive = (request.end_date + timedelta(days=1)).isoformat()
    try:
        asset_prices = fetch_prices(tickers, start, end_exclusive, fill_missing=False)
        benchmark_prices = fetch_prices(
            [market.benchmark_ticker], start, end_exclusive, fill_missing=False
        )
    except (ValueError, RuntimeError, OSError) as exc:
        logger.warning("Price retrieval failed: %s", exc)
        raise HTTPException(
            502, "株価を取得できませんでした。銘柄コードと取得期間を確認してください。"
        ) from exc

    try:
        result = run_web_analysis(
            asset_prices,
            benchmark_prices[market.benchmark_ticker],
            request.strategy,
            request.start_date,
            request.end_date,
            DashboardSettings(),
        )
    except NoEvaluationData as exc:
        raise HTTPException(
            422,
            "指定期間に比較できる日次リターンがありません。銘柄の取引開始日や株価の取得状況を確認してください。",
        ) from exc
    except (ValueError, RuntimeError) as exc:
        logger.warning("Analysis failed: %s", exc)
        raise HTTPException(
            422, "指定した条件で分析できませんでした。期間や戦略を変更してください。"
        ) from exc

    returns = result.returns
    show_annualized = (
        len(returns) >= MIN_ANNUALIZED_OBSERVATIONS and result.skipped_sessions == 0
    )
    cumulative = returns.apply(wealth_index) - 1.0
    labels = (ANALYSIS_STRATEGY_LABELS[result.strategy], market.benchmark_label)
    return AnalysisResponse(
        requested=request.model_copy(update={"tickers": tickers}),
        applied_strategy=result.strategy,
        benchmark_label=market.benchmark_label,
        benchmark_ticker=market.benchmark_ticker,
        currency=market.currency,
        evaluation=EvaluationWindow(
            start_date=returns.index[0].date(),
            end_date=returns.index[-1].date(),
            observations=len(returns),
        ),
        annualized_metrics_available=show_annualized,
        minimum_annualized_observations=MIN_ANNUALIZED_OBSERVATIONS,
        entry_dates={ticker: day.date() for ticker, day in result.entry_dates.items()},
        unavailable_tickers=result.unavailable,
        skipped_sessions=result.skipped_sessions,
        performance=[
            PerformanceRow(
                label=label,
                period_return=float(cumulative[column].iloc[-1]),
                cagr=_finite_or_none(cagr(series)) if show_annualized else None,
                volatility=_finite_or_none(annualized_volatility(series))
                if show_annualized
                else None,
                sharpe=_finite_or_none(sharpe_ratio(series))
                if show_annualized
                else None,
                max_drawdown=_finite_or_none(max_drawdown(series)),
            )
            for label, (column, series) in zip(labels, returns.items())
        ],
        cumulative_returns=[
            ReturnPoint(date=day.date(), strategy=strategy, benchmark=benchmark)
            for day, strategy, benchmark in cumulative.itertuples()
        ],
    )
