# QuantAlloc

QuantAllocは、日本株・東証ETFで構成されたポートフォリオをPythonで定量分析するプロジェクトです。

## 現在作っているもの

Yahoo Financeから価格データを取得し、指定した配分に基づいて次の指標を計算します。

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

## Current Status

v0.1を開発中です。価格取得、ポートフォリオリターン、基本指標とユニットテストまでを対象としています。

## 実行方法

```bash
pip install -r requirements.txt
python main.py
```

テスト:

```bash
pytest
```
