# QuantAlloc

QuantAllocは、日本株ポートフォリオをPythonで定量分析し、TOPIXや複数の資産配分戦略と比較するプロジェクトです。

松尾研究所が提供する講座で学んだ内容を実践し、理解を深める目的で作成しました。

Current version: **v0.9 Robust Covariance Estimation**

## 主な機能

- Yahoo Financeから日本株・東証ETFの価格データを取得
- CAGR、年率ボラティリティ、Sharpe Ratio、Maximum Drawdownを計算
- TOPIX連動ETFと同じ期間でパフォーマンスを比較
- 相関・共分散、HHI、Effective Number of Assets、Risk Contributionを分析
- Minimum Variance、Maximum Sharpe、Efficient Frontier、Risk Parity / ERCを計算
- Walk-ForwardでWeight Drift、Turnover、取引コスト、Gross / Net Returnを評価
- 21・63・126・252日のリバランス頻度を比較
- Sample CovarianceとLedoit-Wolf ShrinkageのOOS成績・Weight安定性を比較

## 分析条件

```text
Estimation Window : 504 trading days
Holding Period    : 63 trading days
Benchmark         : 1306.T（TOPIX ETF）
Risk-Free Rate    : 0%
Transaction Cost  : 10 bps per turnover
Short Selling     : No
Leverage          : No
Covariance        : Sample / Ledoit-Wolf
```

期待リターンには過去の日次平均を252取引日で年率化した値を使用します。Risk Parityは期待リターンを使わず、共分散行列から各銘柄のRisk Contributionが均等になる配分を求めます。

Sample Covarianceを既定値としているため、v0.8までのWalk-Forward結果との互換性を維持しています。

## サンプルポートフォリオ

```text
7203.T  Toyota                    10%
6758.T  Sony                      10%
8306.T  MUFG                      10%
9432.T  NTT                       10%
8058.T  Mitsubishi Corporation    10%
7974.T  Nintendo                  10%
3003.T  HULIC                     10%
6501.T  Hitachi                   10%
9983.T  Fast Retailing            10%
4661.T  Oriental Land             10%
```

## 実行方法

```bash
pip install -r requirements.txt
python main.py
```

テスト:

```bash
pytest
```

## ファイル構成

```text
QuantAlloc/
├── src/
│   ├── backtest.py
│   ├── benchmark.py
│   ├── covariance.py
│   ├── data_provider.py
│   ├── diversification.py
│   ├── evaluation.py
│   ├── metrics.py
│   ├── optimization.py
│   ├── portfolio.py
│   ├── risk.py
│   └── risk_parity.py
├── tests/
├── main.py
├── requirements.txt
└── README.md
```
