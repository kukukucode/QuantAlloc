"""Shared, fixed sample settings for the CLI and Streamlit prototype."""

WEIGHTS = {
    "7203.T": 0.10,
    "6758.T": 0.10,
    "8306.T": 0.10,
    "9432.T": 0.10,
    "8058.T": 0.10,
    "7974.T": 0.10,
    "3003.T": 0.10,
    "6501.T": 0.10,
    "9983.T": 0.10,
    "4661.T": 0.10,
}
TICKERS = list(WEIGHTS)
BENCHMARK_TICKER = "1306.T"
START_DATE = "2021-01-01"
END_DATE = "2026-01-01"
ESTIMATION_WINDOW = 504
HOLDING_PERIOD = 63
TRANSACTION_COST_RATE = 0.001
