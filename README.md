# QuantAlloc

QuantAllocは、日本株で構成されたポートフォリオをPythonで定量分析し、TOPIX連動ETFと比較するプロジェクトです。

## できること

- Yahoo Financeから日本株・東証ETFの価格データを取得
- 指定した配分からポートフォリオの日次リターンを計算
- ポートフォリオとTOPIX連動ETFを共通期間で比較
- ポートフォリオとベンチマークの累積リターンを計算

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
7203.T  Toyota  40%
6758.T  Sony    30%
8306.T  MUFG    30%

Benchmark
1306.T  TOPIX ETF
```

## Current Status

v0.2を開発中です。ポートフォリオとTOPIXのリターンを同じ期間に揃え、4つの基本指標で比較できます。

## 実行方法

```bash
pip install -r requirements.txt
python main.py
```

テスト:

```bash
pytest
```
