# QuantAlloc

QuantAllocは、日本株ポートフォリオをPythonで定量分析し、TOPIXや複数の資産配分戦略と比較するプロジェクトです。

松尾研究所が提供する講座で学んだ内容を実践し、理解を深める目的で作成しました。

Current version: **v0.10 Black-Litterman**

## 主な機能

- Yahoo Financeの日本株・東証ETFデータからCAGR、年率Volatility、Sharpe、Max Drawdownを計算し、TOPIXと比較
- 相関・共分散、HHI、Effective Number of Assets、Risk Contributionを分析
- Minimum Variance、Maximum Sharpe、Efficient Frontier、Risk Parity / ERCを計算
- Walk-ForwardでWeight Drift、Turnover、コスト、Gross / Net、21・63・126・252日のリバランス頻度を評価
- Sample CovarianceとLedoit-Wolf ShrinkageのOOS成績・Weight安定性を比較
- Black-Littermanのprior・P/Q/Ω検証・posteriorから配分を求め、Historical Mean / TOPIXと共通OOS期間で比較

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

Historical Meanは過去の日次平均を252取引日で年率化します。Black-Littermanは各学習期間の共分散とmarket weightsからpriorを計算し、viewを反映したposteriorで配分を求めます。Risk Parityは共分散のみを使用します。

BLのサンプルは**仮定の例**です。market weightsは下記の10銘柄均等配分で、時価総額データは取得しません。viewはToyota − Sonyの年率リターン差2%、δ=2.5、τ=0.05、Ω=0.0004です。Qは年率の総リターンに対するview、Ωはその不確実性の共分散です。

BL入力は固定設定または学習期間だけを受け取る関数で渡せます。market weightsとviewにはリバランス時点で利用可能な情報を使用してください。初回配分のコストは0、Sum Costは各リバランスのコスト率の合計です。TOPIXの売買量・コストは推定せず「-」で表示します。

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
python -m pip install -r requirements.txt
python main.py
```

テスト:

```bash
python -m pytest -q
```

## ファイル構成

```text
QuantAlloc/
├── src/
│   ├── backtest.py
│   ├── benchmark.py
│   ├── black_litterman.py
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
