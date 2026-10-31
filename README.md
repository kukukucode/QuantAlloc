# QuantAlloc

QuantAllocは、日本株ポートフォリオをPythonで定量分析し、TOPIXや複数の資産配分戦略と比較するプロジェクトです。

松尾研究所が提供する講座で学んだ内容を実践し、理解を深める目的で作成しました。

Current version: **v0.19 Factor Exposure Analysis**

## 主な機能

- Yahoo Financeの日本株・東証ETFデータからCAGR、年率Volatility、Sharpe、Max Drawdownを計算し、TOPIXと比較
- 相関・共分散、HHI、Effective Number of Assets、Risk Contributionを分析
- Minimum Variance、Maximum Sharpe、Efficient Frontier、Risk Parity / ERCを計算
- Walk-ForwardでWeight Drift、Turnover、コスト、Gross / Net、21・63・126・252日のリバランス頻度を評価
- Sample CovarianceとLedoit-Wolf ShrinkageのOOS成績・Weight安定性を比較
- Black-Littermanのprior・P/Q/Ω検証・posteriorから配分を求め、Historical Mean / TOPIXと共通OOS期間で比較
- 各戦略とTOPIXのNet累積リターン・Drawdown、および戦略ごとのtarget weight推移をPNGで保存
- Streamlitで4戦略を選択し、TOPIXとの指標・累積リターン・Drawdown・target weightを表示

- i.i.d. Bootstrapによる固定配分ポートフォリオのCAGR・Volatility・Sharpe・Max Drawdownの頑健性分析

- Historical VaR 95%・CVaR 95%による日次下方リスク分析。Streamlitで選択戦略とTOPIXを共通OOS期間で比較

- Streamlitで4戦略とTOPIXのCAGR・Volatility・Sharpe・Max Drawdown・VaR・CVaRを一括比較し、累積リターンとDrawdownを同一グラフで表示

- 63・126・252営業日のRolling Volatility・Rolling Sharpeを計算し、Streamlitで4戦略とTOPIXのリスク変化を時系列比較

- TOPIXの21営業日累積リターンが最も低い共通OOS期間を抽出し、4戦略とTOPIXの期間リターン・最大Drawdown・回復日数を比較

- TOPIXを市場ファクターとする日次単回帰でMarket Beta・Annual Alpha・R²を推定し、Streamlitで4戦略を比較

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

Bootstrapは日次リターンの行を復元抽出し、固定配分で1,000回評価します。2.5%点・中央値・97.5%点を表示します。これはin-sample分析であり、将来の成績予測ではありません。

VaR・CVaRは共通OOS期間の日次リターンから推定します。選択戦略は売買コスト控除後のNet、TOPIXはコスト未控除です。CVaRは最悪5%の部分における平均損失を表し、観測数に応じて端数の重みを使用します。いずれも過去データに基づく推定で、将来の損失上限ではありません。

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

`outputs/charts/`にNet累積リターン、Drawdown、5戦略の配分ヒートマップを保存します。
成績グラフは共通OOS期間の日次リターンを複利計算し、TOPIXは取得したベンチマークのリターンをそのまま使用します（売買コストは推定しません）。
配分ヒートマップは全リバランスのtarget weightで、日々driftした保有比率ではありません。GUIは不要です。

Streamlit画面:

```bash
python -m streamlit run app.py
```

Equal Weight／Minimum Variance／Maximum Sharpe／Risk Parityから戦略を選び、「分析を実行」で取得・計算します。銘柄・期間・学習／保有日数・売買コストは固定で、最適化にはSample共分散を使います。BLは画面では選択できません。
価格は取得条件ごとに1時間キャッシュし、戦略切り替えでは再取得しません。選択を変えると前の結果を消し、再実行を待ちます。最適化失敗時に別戦略へ切り替えることはありません。

Streamlitの「4戦略を比較」では、Equal Weight、Minimum Variance、Maximum Sharpe、Risk Parity、TOPIXを同一OOS期間で比較します。4戦略は売買コスト控除後のNetリターン、TOPIXは売買コスト未控除のリターンを使用します。従来の1戦略選択・分析機能も利用できます。

Rolling Risk Analysisでは、共通OOS期間の日次リターンから63・126・252営業日の年率VolatilityとSharpeを計算します。年率換算は252営業日、リスクフリーレートは0%です。戦略には売買コスト控除後のNetリターンを使用します。観測数が足りない期間やVolatilityが実質0の期間では、計算できない指標を欠損値として扱います。

Stress Testingでは、共通OOS期間のTOPIX日次リターンから、21営業日累積リターンが最も低い期間を事後的に抽出します。その同じ期間の4戦略とTOPIXについて、累積リターン、最大Drawdown、回復日数を比較します。回復日数はストレス期間終了後、期間開始前の資産価値を回復するまでの営業日数です。期間終了時点で回復済みなら0日、観測期間中に回復しなければ未回復とします。4戦略は売買コスト控除後、TOPIXは売買コスト未控除です。この分析は過去の厳しい市場局面を振り返るもので、将来の暴落を予測するものではありません。

Factor Exposure Analysisでは、共通OOS期間の4戦略とTOPIXの日次リターンを使用し、TOPIXを説明変数とする単一市場ファクター回帰を行います。Market Betaは市場への感応度、Annual Alphaは日次回帰切片を252営業日で算術年率化した値、R²は市場モデルによる変動の説明割合です。4戦略には売買コスト控除後のNetリターン、TOPIXには売買コスト未控除のリターンを使用します。市場リターンの分散が実質0の場合、指標は算出不可です。定数リターン系列のR²も算出不可とします。Alphaの統計的有意性は検定しておらず、因果関係や将来の超過収益を示すものではありません。

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
│   ├── dashboard.py
│   ├── diversification.py
│   ├── evaluation.py
│   ├── factor.py
│   ├── metrics.py
│   ├── optimization.py
│   ├── portfolio.py
│   ├── risk.py
│   ├── risk_parity.py
│   ├── rolling.py
│   ├── sample_config.py
│   ├── stress.py
│   └── visualization.py
├── tests/
├── app.py
├── main.py
├── requirements.txt
└── README.md
```
