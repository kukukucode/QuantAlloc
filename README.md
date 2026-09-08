# QuantAlloc

QuantAllocは、日本株で構成されたポートフォリオをPythonで定量分析し、TOPIXとの比較や分散状況を確認するプロジェクトです。

## できること

- Yahoo Financeから日本株・東証ETFの価格データを取得
- 指定した配分からポートフォリオの日次リターンを計算
- ポートフォリオとTOPIX連動ETFを共通期間で比較
- ポートフォリオとベンチマークの累積リターンを計算
- 各銘柄の日次リターンから相関行列を計算
- 日次共分散行列を252取引日で年率化
- HHIとEffective Number of Assetsで配分の集中度を確認
- 各銘柄のRisk Contributionを計算
- Minimum Variance Portfolioを計算
- Maximum Sharpe Portfolioを計算
- Efficient Frontierを計算
- 3戦略をrolling walk-forward方式でバックテスト
- OOSリターン、weight履歴、学習・評価期間を保存
- 全戦略とTOPIXを共通OOS日付に揃えて公平に比較
- OOS Strategy Comparisonを1つの表として出力
- 保有期間中のweight driftを考慮してTurnoverを計算
- 戦略別の平均・累積・最大Turnoverを比較
- Turnoverに取引コストを適用してGross / Net Returnを保存
- 21・63・126・252日のrebalance頻度を比較

比較する指標:

- CAGR
- 年率ボラティリティ
- Sharpe Ratio
- Maximum Drawdown

## 使用技術

- Python
- NumPy
- pandas
- yfinance
- SciPy
- pytest

## Expected Returnの前提

最適化で使用する期待リターンは、過去の日次平均リターンを252取引日で年率化したhistorical mean returnです。将来のリターンを予測するものではありません。

最適化では次の制約を使用します。

- 空売りなし
- レバレッジなし
- 配分合計100%

## Walk-Forward Backtest

過去データだけでweightを決め、その後の未知期間で運用成績を評価します。

```text
Estimation Window : 504 trading days
Holding Period    : 63 trading days
Window Type       : Rolling
Strategies        : Equal Weight
                    Minimum Variance
                    Maximum Sharpe
Risk-Free Rate    : 0%
Short Selling     : No
Leverage          : No
Transaction Cost  : 10 bps per turnover
```

完全な63日OOS windowだけを採用し、同じOOS日付のTOPIXと比較します。

## Realistic Backtesting

前回のtarget weightを保有期間中の各資産リターンで変化させ、rebalance直前のweightを復元します。そのweightと次のtarget weightの差の絶対値を合計してTurnoverを計算します。

```text
Turnover = sum(abs(new target weight - pre-rebalance weight))
```

初回のTurnoverは0とし、各戦略についてrebalance別Turnover、平均Turnover、累積Turnover、最大Turnoverを出力します。

```text
Transaction Cost = Turnover * Cost Rate
```

既定のCost Rateは10bpsです。コスト控除前のGross Returnと控除後のNet Returnを両方保存します。

同じ戦略を21日、63日、126日、252日のholding periodで実行し、共通OOS期間のNet CAGR、Net Sharpe、平均Turnoverを比較します。

## サンプル構成

```text
Portfolio
7203.T  Toyota                    Stock  10%
6758.T  Sony                      Stock  10%
8306.T  MUFG                      Stock  10%
9432.T  NTT                       Stock  10%
8058.T  Mitsubishi Corporation    Stock  10%
7974.T  Nintendo                  Stock  10%
3003.T  HULIC                     Stock  10%
6501.T  Hitachi                   Stock  10%
9983.T  Fast Retailing            Stock  10%
4661.T  Oriental Land             Stock  10%

Benchmark
1306.T  TOPIX ETF
```

## Current Status

現在はv0.7 Realistic Backtestingです。weight drift、Turnover、取引コスト、Gross / Net Return、rebalance頻度を含むOOS比較を実行できます。

## 実行方法

```bash
pip install -r requirements.txt
python main.py
```

テスト:

```bash
pytest
```
