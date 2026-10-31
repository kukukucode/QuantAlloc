"""Streamlit strategy selection with a fixed asset universe and settings."""

import pandas as pd
import streamlit as st

from src.dashboard import (
    SampleAnalysis,
    StrategyComparison,
    run_sample_analysis,
    run_sample_sensitivity,
    run_strategy_comparison,
)
from src.dashboard_settings import DashboardSettings
from src.data_provider import fetch_prices
from src.export import create_export_archive
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

    st.subheader("Stress Testing（Worst TOPIX 21-day Window）")
    st.caption(
        f"TOPIXが最も低い21営業日リターンを記録した期間: "
        f"{result.stress.start.date()} → {result.stress.end.date()}"
    )
    st.caption(
        "共通OOS期間から事後的に抽出した局面です。"
        "回復日数はストレス期間終了後、期間開始前の資産価値を"
        "再び上回るまでの営業日数です。"
        "未回復は観測期間内に回復が確認できないことを示します。"
    )

    stress_table = result.stress.summary.rename(
        columns={
            "window_return": "Stress Return",
            "max_drawdown": "Stress Max Drawdown",
            "recovery_days": "Recovery Days",
        }
    )
    stress_table.attrs = {}

    st.dataframe(
        stress_table.style.format(
            {
                "Stress Return": "{:.2%}",
                "Stress Max Drawdown": "{:.2%}",
                "Recovery Days": "{:.0f}",
            },
            na_rep="未回復",
        ),
        width="stretch",
    )

    st.line_chart(
        result.stress.cumulative_returns * 100.0,
        x_label="Date",
        y_label="Stress cumulative return (%)",
        height=400,
    )

    st.subheader("Factor Exposure（TOPIX Market Model）")
    st.caption(
        "共通OOS期間の日次リターンを使った単回帰分析です。"
        "BetaはTOPIXへの感応度、Annual Alphaは日次切片の"
        "252営業日換算、R²は市場モデルの説明力です。"
    )
    st.caption(
        "4戦略は売買コスト控除後のNet、TOPIXはコスト未控除。"
        "Alphaは将来の超過収益や運用スキルを保証しません。"
        "市場変動がない場合などは算出不可です。"
    )

    factors = result.factor_exposure.rename(
        columns={
            "beta": "Market Beta",
            "alpha_annual": "Annual Alpha",
            "r_squared": "R²",
        }
    )
    factors.attrs = {}

    st.dataframe(
        factors.style.format(
            {
                "Market Beta": "{:.3f}",
                "Annual Alpha": "{:.2%}",
                "R²": "{:.3f}",
            },
            na_rep="算出不可",
        ),
        width="stretch",
    )

    betas = result.factor_exposure["beta"].drop(index="TOPIX").dropna()

    if not betas.empty:
        st.bar_chart(
            betas.rename("Market Beta"),
            x_label="Strategy",
            y_label="Market Beta",
        )


def display_sensitivity(result) -> None:
    """Show three independent parameter sweeps."""
    st.subheader(f"Parameter Sensitivity — {STRATEGY_LABELS[result.strategy]}")
    st.caption(
        "選択戦略を、サイドバーで指定した条件を基準に再バックテストします。"
        "学習期間・保有期間・売買コストを1条件ずつ変更します。"
        "各表の成績はその表内の共通OOS期間で比較します。"
    )
    st.caption(
        "CAGRなどは売買コスト控除後のNet成績です。"
        "Total Costは共通OOS期間内のリバランスコスト率の合計です。"
        "異なる表の数値は評価期間が違う場合があります。"
    )

    labels = {
        "estimation_window": "学習期間（営業日）",
        "holding_period": "リバランス頻度（営業日）",
        "transaction_cost_rate": "売買コスト率",
    }

    for parameter, label in labels.items():
        table = result.tables[parameter]
        start = table.attrs["start_date"].date()
        end = table.attrs["end_date"].date()

        st.subheader(label)
        st.caption(
            f"共通OOS期間: {start} → {end} / {table.attrs['observations']} observations"
        )

        display = table.rename(
            columns={
                "cagr": "CAGR",
                "volatility": "Volatility",
                "sharpe": "Sharpe",
                "max_drawdown": "Max Drawdown",
                "total_transaction_cost": "Total Cost",
                "status": "Status",
            }
        )
        display.attrs = {}

        st.dataframe(
            display.style.format(
                {
                    "CAGR": "{:.2%}",
                    "Volatility": "{:.2%}",
                    "Sharpe": "{:.2f}",
                    "Max Drawdown": "{:.2%}",
                    "Total Cost": "{:.2%}",
                },
                na_rep="算出不可",
            ),
            width="stretch",
        )


def display_export(
    result,
    settings: DashboardSettings,
    price_request: tuple,
) -> None:
    """Export a saved result and the input prices used to produce it."""
    asset_prices, benchmark_prices = st.session_state["sample_export_inputs"]

    archive = create_export_archive(
        result,
        settings,
        asset_prices,
        benchmark_prices,
        start_date=price_request[2],
        end_date=price_request[3],
        benchmark_ticker=price_request[1],
    )

    st.download_button(
        "結果と設定をZIPで保存",
        data=archive,
        file_name="quantalloc_v022_analysis.zip",
        mime="application/zip",
        key="download_analysis_export",
    )


def main() -> None:
    st.set_page_config(page_title="QuantAlloc", page_icon="📊", layout="wide")
    st.title("QuantAlloc")
    st.caption("v0.23 Archive Verification & Replay")
    st.write(
        "既存10銘柄・選択した条件で配分戦略を選び、TOPIXとWalk-Forward成績を比較します。"
    )
    with st.sidebar:
        st.header("分析条件")
        strategy = st.selectbox(
            "戦略",
            list(STRATEGY_LABELS),
            format_func=STRATEGY_LABELS.__getitem__,
            key="selected_strategy",
        )
        st.caption("設定を変更すると、前回の分析結果は非表示になります。")

        training_options = (252, 378, 504)
        holding_options = (21, 63, 126, 252)
        cost_options = (0, 10, 30)

        training = st.selectbox(
            "学習期間（営業日）",
            training_options,
            index=training_options.index(ESTIMATION_WINDOW),
            key="selected_training",
        )

        holding = st.selectbox(
            "リバランス頻度（営業日）",
            holding_options,
            index=holding_options.index(HOLDING_PERIOD),
            key="selected_holding",
        )

        cost_bps = st.selectbox(
            "売買コスト（bps）",
            cost_options,
            index=cost_options.index(int(TRANSACTION_COST_RATE * 10000)),
            key="selected_cost",
        )

        settings = DashboardSettings(
            estimation_window=training,
            holding_period=holding,
            transaction_cost_rate=cost_bps / 10000,
        )

        st.write(f"Benchmark: {BENCHMARK_TICKER}（TOPIX ETF）")
        st.write("Covariance: Sample（最適化戦略）")
        st.write("Risk-free rate: 0% / No short / No leverage")
        st.caption(f"取得期間: {START_DATE} ～ {END_DATE}（終了日を含まない）")
        st.subheader("対象銘柄")
        st.dataframe(pd.DataFrame({"Ticker": TICKERS}), hide_index=True)

    st.caption("価格データは1時間キャッシュします。初回配分の売買コストは0です。")
    price_request = (tuple(TICKERS), BENCHMARK_TICKER, START_DATE, END_DATE)
    analysis_request = (price_request, strategy, settings)
    comparison_request = (price_request, settings)
    if st.session_state.get("sample_analysis_request") != analysis_request:
        st.session_state.pop("sample_analysis", None)
    if st.session_state.get("sample_comparison_request") != comparison_request:
        st.session_state.pop("sample_comparison", None)
    if st.session_state.get("sample_sensitivity_request") != analysis_request:
        st.session_state.pop("sample_sensitivity", None)
    if not any(
        key in st.session_state
        for key in ("sample_analysis", "sample_comparison", "sample_sensitivity")
    ):
        st.session_state.pop("sample_export_inputs", None)

    if st.button("分析を実行", type="primary"):
        # Never present a previous successful result after a failed new run.
        st.session_state.pop("sample_analysis", None)
        st.session_state.pop("sample_comparison", None)
        st.session_state.pop("sample_sensitivity", None)
        st.session_state.pop("sample_export_inputs", None)
        try:
            with st.spinner("価格データを取得し、分析しています…"):
                asset_prices, benchmark_prices = load_sample_prices(*price_request)
                if settings == DashboardSettings():
                    analysis = run_sample_analysis(
                        asset_prices, benchmark_prices, strategy
                    )
                else:
                    analysis = run_sample_analysis(
                        asset_prices,
                        benchmark_prices,
                        strategy,
                        settings=settings,
                    )

                st.session_state["sample_analysis"] = analysis
                st.session_state["sample_analysis_request"] = analysis_request
                st.session_state["sample_export_inputs"] = (
                    asset_prices.copy(),
                    benchmark_prices.copy(),
                )
        except (ValueError, TypeError, KeyError, RuntimeError, OSError) as exc:
            st.error(f"分析できませんでした: {exc}")

    if st.button("4戦略を比較"):
        st.session_state.pop("sample_analysis", None)
        st.session_state.pop("sample_comparison", None)
        st.session_state.pop("sample_sensitivity", None)
        st.session_state.pop("sample_export_inputs", None)
        try:
            with st.spinner("4戦略をバックテストしています…"):
                asset_prices, benchmark_prices = load_sample_prices(*price_request)
                st.session_state["sample_comparison"] = run_strategy_comparison(
                    asset_prices, benchmark_prices, settings=settings
                )
                st.session_state["sample_comparison_request"] = comparison_request
                st.session_state["sample_export_inputs"] = (
                    asset_prices.copy(),
                    benchmark_prices.copy(),
                )
        except (ValueError, TypeError, KeyError, RuntimeError, OSError) as exc:
            st.error(f"比較できませんでした: {exc}")

    if st.button("感応度を分析"):
        st.session_state.pop("sample_analysis", None)
        st.session_state.pop("sample_comparison", None)
        st.session_state.pop("sample_sensitivity", None)
        st.session_state.pop("sample_export_inputs", None)
        try:
            with st.spinner("パラメータ感応度を分析しています…"):
                asset_prices, benchmark_prices = load_sample_prices(*price_request)
                st.session_state["sample_sensitivity"] = run_sample_sensitivity(
                    asset_prices, strategy, settings=settings
                )
                st.session_state["sample_sensitivity_request"] = analysis_request
                st.session_state["sample_export_inputs"] = (
                    asset_prices.copy(),
                    benchmark_prices.copy(),
                )
        except (ValueError, TypeError, KeyError, RuntimeError, OSError) as exc:
            st.error(f"感応度を分析できませんでした: {exc}")

    if "sample_sensitivity" in st.session_state:
        display_sensitivity(st.session_state["sample_sensitivity"])
        display_export(st.session_state["sample_sensitivity"], settings, price_request)
        return

    if "sample_comparison" in st.session_state:
        display_comparison(st.session_state["sample_comparison"])
        display_export(st.session_state["sample_comparison"], settings, price_request)
        return

    if "sample_analysis" not in st.session_state:
        st.info("分析を実行、または4戦略を比較してください。")
        return

    display_analysis(st.session_state["sample_analysis"])
    display_export(st.session_state["sample_analysis"], settings, price_request)


if __name__ == "__main__":
    main()
