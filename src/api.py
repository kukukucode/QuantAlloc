"""Local HTTP interface for the single-strategy QuantAlloc dashboard."""

import logging
import math
from datetime import date, timedelta

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, FiniteFloat

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
app = FastAPI(title="QuantAlloc", version="0.27")


class DateRange(BaseModel):
    start_date: date
    end_date: date


class AnalysisRequest(DateRange):
    model_config = ConfigDict(extra="forbid")

    strategy: str


class AnalysisConfig(BaseModel):
    defaults: AnalysisRequest
    strategies: dict[str, str]
    benchmark_ticker: str
    asset_count: int
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
    evaluation: EvaluationWindow
    performance: list[PerformanceRow]
    cumulative_returns: list[ReturnPoint]


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


@app.get("/api/config")
def configuration() -> AnalysisConfig:
    settings = DashboardSettings()
    return AnalysisConfig(
        defaults=AnalysisRequest(
            strategy="equal_weight",
            start_date=date.fromisoformat(START_DATE),
            end_date=date.fromisoformat(END_DATE) - timedelta(days=1),
        ),
        strategies=ANALYSIS_STRATEGY_LABELS,
        benchmark_ticker=BENCHMARK_TICKER,
        asset_count=len(TICKERS),
        estimation_window=settings.estimation_window,
        holding_period=settings.holding_period,
        transaction_cost_bps=round(settings.transaction_cost_rate * 10_000),
        maximum_date=date.today(),
    )


@app.post("/api/analysis")
def analyze(request: AnalysisRequest) -> AnalysisResponse:
    """Analyze inclusive dates using the same prices and engine as Streamlit."""
    if request.strategy not in ANALYSIS_STRATEGY_LABELS:
        raise HTTPException(422, "選択できる戦略を指定してください。")
    if not request.start_date <= request.end_date <= date.today():
        raise HTTPException(422, "終了日は開始日以降かつ今日以前にしてください。")

    start = request.start_date.isoformat()
    end_exclusive = (request.end_date + timedelta(days=1)).isoformat()
    try:
        asset_prices = fetch_prices(TICKERS, start, end_exclusive)
        benchmark_prices = fetch_prices([BENCHMARK_TICKER], start, end_exclusive)
    except (ValueError, RuntimeError, OSError) as exc:
        logger.warning("Price retrieval failed: %s", exc)
        raise HTTPException(
            502, "株価を取得できませんでした。取得期間を確認し、再度お試しください。"
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
            asset_prices, benchmark_prices, request.strategy, settings=settings
        )
    except (ValueError, RuntimeError) as exc:
        logger.warning("Analysis failed: %s", exc)
        raise HTTPException(
            422, "指定した条件で分析できませんでした。期間や戦略を変更してください。"
        ) from exc

    summary = result.performance
    return AnalysisResponse(
        requested=request,
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
