"""Streamlit strategy selection with a fixed asset universe and settings."""

import pandas as pd
import streamlit as st

from src.dashboard import (
    SampleAnalysis,
    StrategyComparison,
    run_sample_analysis,
    run_strategy_comparison,
)
from src.data_provider import fetch_prices
from src.sample_config import (
    BENCHMARK_TICKER,
    END_DATE,
    ESTIMATION_WINDOW,
    HOLDING_PERIOD,
    START_DATE,
    STRATEGY_LABELS,
    TICKERS,
    TRANSACTION_COST_RATE,
)


@st.cache_data(ttl=3600, show_spinner=False)
def load_sample_prices(
    tickers: tuple[str, ...], benchmark_ticker: str, start: str, end: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cache downloads by their explicit request, not by mutable globals."""
    return (
        fetch_prices(list(tickers), start, end),
        fetch_prices([benchmark_ticker], start, end),
    )


def display_analysis(result: SampleAnalysis) -> None:
    """Render metrics and existing chart data without rerunning a backtest."""
    summary = result.performance
    strategy_label = STRATEGY_LABELS[result.strategy]
    start = summary.attrs["start_date"].date()
    end = summary.attrs["end_date"].date()
    st.caption(
        f"共通OOS期間: {start} → {end} / {summary.attrs['observations']} observations"
    )
    values = summary.loc[strategy_label]
    columns = st.columns(4)
    for column, key, label in zip(
        columns,
        ("cagr", "volatility", "sharpe", "max_drawdown"),
        ("Net CAGR", "Volatility", "Sharpe", "Max Drawdown"),
    ):
        value = values[key]
        value_text = (
            "算出不可"
            if pd.isna(value)
            else f"{value:.2f}"
            if key == "sharpe"
            else f"{value:.2%}"
        )
        column.metric(label, value_text)

    st.subheader(f"{strategy_label} vs TOPIX")
    display_summary = summary.rename(
        columns={
            "cagr": "CAGR",
            "volatility": "Volatility",
            "sharpe": "Sharpe",
            "max_drawdown": "Max Drawdown",
        }
    )
    # Period metadata is retained on the result, not sent through Arrow's JSON metadata.
    display_summary.attrs = {}
    st.dataframe(
        display_summary.style.format(
            {
                "CAGR": "{:.2%}",
                "Volatility": "{:.2%}",
                "Sharpe": "{:.2f}",
                "Max Drawdown": "{:.2%}",
            },
            na_rep="算出不可",
        ),
        width="stretch",
    )
    st.caption(
        f"{strategy_label}は売買コスト控除後のNet成績です。"
        "TOPIX（1306.T）は取得リターンをそのまま使用し、売買コストは推定しません。"
    )
    st.subheader("Tail Risk（Historical / Daily / 95%）")
    st.caption(
        "共通OOS期間の日次リターンから推定した損失率です。"
        "選択戦略は売買コスト控除後のNet、TOPIXはコスト未控除です。"
        "数値が大きいほど下方リスクが大きいことを示します。"
    )
    risk_table = result.tail_risk.rename(
        columns={"var_95": "VaR 95%", "cvar_95": "CVaR 95%"}
    )
    risk_table.attrs = {}
    st.dataframe(
        risk_table.style.format({"VaR 95%": "{:.2%}", "CVaR 95%": "{:.2%}"}),
        width="stretch",
    )

    cumulative, drawdown, allocation = st.tabs(["累積リターン", "Drawdown", "配分"])
    with cumulative:
        st.line_chart(
            result.cumulative_returns * 100.0,
            x_label="Date",
            y_label="Cumulative return (%)",
            height=400,
        )
    with drawdown:
        st.line_chart(
            result.drawdowns * 100.0,
            x_label="Date",
            y_label="Drawdown (%)",
            height=400,
        )
    with allocation:
        st.caption(
            "各リバランスのtarget weightです。日々driftした保有比率ではありません。"
        )
        st.bar_chart(
            (result.backtest.target_weights.iloc[-1] * 100.0).rename(
                "Target weight (%)"
            ),
            x_label="Asset",
            y_label="Latest target weight (%)",
        )
        st.dataframe(
            result.backtest.target_weights.style.format("{:.1%}"), width="stretch"
        )


def display_comparison(result: StrategyComparison) -> None:
    """Display five return series without rerunning the backtests."""
    summary = result.performance
    start = summary.attrs["start_date"].date()
    end = summary.attrs["end_date"].date()

    st.subheader("4戦略とTOPIXの一括比較")
    st.caption(
        f"共通OOS期間: {start} → {end} / {summary.attrs['observations']} observations"
    )
    st.caption("4戦略は売買コスト控除後のNet成績です。TOPIXは売買コスト未控除です。")

    performance = summary.rename(
        columns={
            "cagr": "CAGR",
            "volatility": "Volatility",
            "sharpe": "Sharpe",
            "max_drawdown": "Max Drawdown",
        }
    )
    performance.attrs = {}

    st.dataframe(
        performance.style.format(
            {
                "CAGR": "{:.2%}",
                "Volatility": "{:.2%}",
                "Sharpe": "{:.2f}",
                "Max Drawdown": "{:.2%}",
            },
            na_rep="算出不可",
        ),
        width="stretch",
    )

    st.subheader("Tail Risk（Historical / Daily / 95%）")
    risk = result.tail_risk.rename(
        columns={
            "var_95": "VaR 95%",
            "cvar_95": "CVaR 95%",
        }
    )
    risk.attrs = {}

    st.dataframe(
        risk.style.format(
            {
                "VaR 95%": "{:.2%}",
                "CVaR 95%": "{:.2%}",
            }
        ),
        width="stretch",
    )

    cumulative_tab, drawdown_tab = st.tabs(["累積リターン", "Drawdown"])

    with cumulative_tab:
        st.line_chart(
            result.cumulative_returns * 100.0,
            x_label="Date",
            y_label="Cumulative return (%)",
            height=400,
        )

    with drawdown_tab:
        st.line_chart(
            result.drawdowns * 100.0,
            x_label="Date",
            y_label="Drawdown (%)",
            height=400,
        )

    st.subheader("Rolling Risk Analysis")
    st.caption(
        "共通OOS期間から計算した年率VolatilityとSharpeです。"
        "リスクフリーレートは0%です。"
    )

    window = st.selectbox(
        "ローリング期間",
        (63, 126, 252),
        format_func=lambda days: f"{days}営業日",
        key="rolling_window",
    )

    volatility = result.rolling_volatility[window].dropna(how="all")
    sharpe = result.rolling_sharpe[window].dropna(how="all")

    if volatility.empty:
        st.info("この期間では観測数が不足しています。")
    else:
        volatility_tab, sharpe_tab = st.tabs(["Rolling Volatility", "Rolling Sharpe"])

        with volatility_tab:
            st.line_chart(
                volatility * 100.0,
                x_label="Date",
                y_label="Annualized volatility (%)",
                height=400,
            )

        with sharpe_tab:
            if sharpe.empty:
                st.info("Sharpeを算出できる期間がありません。")
            else:
                st.line_chart(
                    sharpe,
                    x_label="Date",
                    y_label="Rolling Sharpe",
                    height=400,
                )


def main() -> None:
    st.set_page_config(page_title="QuantAlloc", page_icon="📊", layout="wide")
    st.title("QuantAlloc")
    st.caption("v0.17 Rolling Risk Analysis")
    st.write(
        "既存10銘柄・固定条件で配分戦略を選び、TOPIXとWalk-Forward成績を比較します。"
    )
    with st.sidebar:
        st.header("分析条件")
        strategy = st.selectbox(
            "戦略",
            list(STRATEGY_LABELS),
            format_func=STRATEGY_LABELS.__getitem__,
            key="selected_strategy",
        )
        st.caption("戦略以外の条件は固定です。")
        st.write(f"Benchmark: {BENCHMARK_TICKER}（TOPIX ETF）")
        st.write(f"Training: {ESTIMATION_WINDOW} trading days")
        st.write(f"Holding: {HOLDING_PERIOD} trading days")
        st.write(f"Cost: {TRANSACTION_COST_RATE * 10000:g} bps per turnover")
        st.write("Covariance: Sample（最適化戦略）")
        st.write("Risk-free rate: 0% / No short / No leverage")
        st.caption(f"取得期間: {START_DATE} ～ {END_DATE}（終了日を含まない）")
        st.subheader("対象銘柄")
        st.dataframe(pd.DataFrame({"Ticker": TICKERS}), hide_index=True)

    st.caption("価格データは1時間キャッシュします。初回配分の売買コストは0です。")
    price_request = (tuple(TICKERS), BENCHMARK_TICKER, START_DATE, END_DATE)
    analysis_request = (price_request, strategy)
    if st.session_state.get("sample_analysis_request") != analysis_request:
        st.session_state.pop("sample_analysis", None)
    if st.session_state.get("sample_comparison_request") != price_request:
        st.session_state.pop("sample_comparison", None)
    if st.button("分析を実行", type="primary"):
        # Never present a previous successful result after a failed new run.
        st.session_state.pop("sample_analysis", None)
        st.session_state.pop("sample_comparison", None)
        try:
            with st.spinner("価格データを取得し、分析しています…"):
                asset_prices, benchmark_prices = load_sample_prices(*price_request)
                st.session_state["sample_analysis"] = run_sample_analysis(
                    asset_prices, benchmark_prices, strategy
                )
                st.session_state["sample_analysis_request"] = analysis_request
        except (ValueError, TypeError, KeyError, RuntimeError, OSError) as exc:
            st.error(f"分析できませんでした: {exc}")

    if st.button("4戦略を比較"):
        st.session_state.pop("sample_analysis", None)
        st.session_state.pop("sample_comparison", None)
        try:
            with st.spinner("4戦略をバックテストしています…"):
                asset_prices, benchmark_prices = load_sample_prices(*price_request)
                st.session_state["sample_comparison"] = run_strategy_comparison(
                    asset_prices, benchmark_prices
                )
                st.session_state["sample_comparison_request"] = price_request
        except (ValueError, TypeError, KeyError, RuntimeError, OSError) as exc:
            st.error(f"比較できませんでした: {exc}")

    if "sample_comparison" in st.session_state:
        display_comparison(st.session_state["sample_comparison"])
        return

    if "sample_analysis" not in st.session_state:
        st.info("分析を実行、または4戦略を比較してください。")
        return

    display_analysis(st.session_state["sample_analysis"])


if __name__ == "__main__":
    main()
