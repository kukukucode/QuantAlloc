
# QuantAlloc

日本株の資産配分を最適化し、TOPIXと運用成績を比較する定量分析アプリです。Pythonを中心に、一部の計算をRustで高速化しています。

開発版：v0.24（Python・Rustハイブリッド）

## 主な機能

- 最小分散・最大シャープレシオ・リスク均等配分・ブラック・リッターマンによる資産配分
- 時系列バックテストとTOPIX比較（収益率・リスク・売買コスト）
- 共分散推定、損失リスク・ストレス・感応度分析
- Streamlitでの戦略比較、条件変更、グラフ表示
- 分析結果のZIP保存、整合性検証、再計算

## 分析条件

初期設定は日本株10銘柄、学習期間504営業日、リバランス間隔63営業日、売買コスト10bpsです。ベンチマークにはTOPIX連動ETF（1306.T）を使用します。空売り・レバレッジは使用しません。

Streamlit上では学習期間・リバランス間隔・売買コストを変更できます。銘柄と株価取得期間は現在固定です。

## 実行方法

Python 3.10以降を使用します。

```bash
python -m pip install -r requirements.txt

# 分析結果をグラフとして保存
python main.py

# ブラウザで分析
python -m streamlit run app.py

# テスト
python -m pytest -q
```

## Rustによる高速化（任意）

Pythonがデータ取得・最適化・結果表示を担当し、Rustが保有期間の損益と保有比率の変化を計算します。Rustがなくても動作します。最大シャープレシオ最適化には解析的な勾配を使用します。

Rustツールチェーンを導入済みの場合：

```bash
python -m pip install "maturin>=1.10,<2"
python -m maturin build --release --manifest-path rust/Cargo.toml --out dist
python -m pip install dist/*.whl

# Rustとの数値一致テスト
QUANTALLOC_ENGINE=rust python -m pytest -q tests/test_fast_backend.py tests/test_backtest.py

# 処理速度の比較
python -m benchmarks.holding_kernel
```

実行エンジンは環境変数 `QUANTALLOC_ENGINE` で `auto`（標準）・`python`・`rust` を選択できます。高速化率は処理内容や環境に依存します。

## 保存データの検証

画面から出力したZIPは、通信なしで整合性の確認とバックテストの再計算ができます。

```bash
# チェックサム・ファイル構成を検証
python -m src.export_verify quantalloc_v022_analysis.zip --integrity-only

# 保存データで分析を再計算
python -m src.export_verify quantalloc_v022_analysis.zip
```

## 注意事項

株価はYahoo Financeから取得します。分析結果は過去データに基づくもので、将来の運用成績を保証しません。TOPIX側の売買コストは控除していません。ブラック・リッターマンのサンプル見通しは説明用の仮定です。ZIPのハッシュ検証はデータの真正性を証明するものではありません。
