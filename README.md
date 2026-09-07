# QuantAlloc

QuantAlloc は、株式・ETFで構成されたポートフォリオを定量的に分析するための金融工学プロジェクトです。

現在は、Pythonでポートフォリオ分析の基礎となる計算エンジンを実装しています。

## Current Scope

最初の実装では、銘柄と配分を入力し、過去の価格データから以下を計算します。

- 日次リターン
- ポートフォリオリターン
- CAGR
- 年率ボラティリティ
- Sharpe Ratio
- Maximum Drawdown

例:

AAPL  30%
MSFT  30%
SPY   40%

## Data

市場データの取得には、まず yfinance を使用します。

データ取得処理と金融計算処理は分離して実装します。

## Project Structure

quantalloc/
├── src/
│   ├── data_provider.py
│   ├── portfolio.py
│   └── metrics.py
├── tests/
├── main.py
├── requirements.txt
└── README.md

## Development Environment

- Python 3.12
- GitHub Codespaces
- venv
- pip

使用予定の主なライブラリ:

- NumPy
- pandas
- yfinance
- SciPy
- pytest

## Status

現在は初期実装段階です。

まずは、

価格データ取得
→ 日次リターン
→ ポートフォリオリターン
→ リスク・リターン指標
→ Unit Test

の順で実装します。