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
- pytest

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

v0.3を開発中です。ポートフォリオとTOPIXの比較に加え、相関・共分散と配分集中度を分析できます。

## 実行方法

```bash
pip install -r requirements.txt
python main.py
```

テスト:

```bash
pytest
```
