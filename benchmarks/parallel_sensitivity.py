"""Observe end-to-end serial vs. spawned-process sensitivity time.

Does not set a pass/fail speed threshold. Includes worker startup and
serialization overhead. Run from a file-backed Python process.
"""

from statistics import median
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.sensitivity import run_parameter_sensitivity


def time_run(returns: pd.DataFrame, max_workers: int, repetitions: int = 3) -> float:
    times = []
    for _ in range(repetitions):
        started = perf_counter()
        run_parameter_sensitivity(
            returns,
            "hierarchical_risk_parity",
            estimation_window=504,
            holding_period=63,
            estimation_windows=(252, 378, 504),
            holding_periods=(21, 63, 126),
            transaction_cost_rates=(0.0, 0.001, 0.003),
            max_workers=max_workers,
        )
        times.append(perf_counter() - started)
    return median(times)


def main() -> None:
    rng = np.random.default_rng(101)
    data = pd.DataFrame(
        rng.normal(0.0005, 0.012, size=(900, 24)),
        index=pd.bdate_range("2020-01-01", periods=900),
        columns=[f"A{index:02d}" for index in range(24)],
    )
    with threadpool_limits(limits=1):
        sequential = time_run(data, 1)
        parallel = time_run(data, 2)
    print("24 assets x 900 trading days, HRP sensitivity, 3 median runs")
    print(f"1 process: {sequential:.3f}s")
    print(f"2 processes: {parallel:.3f}s")
    print(f"speed ratio (1/2): {sequential / parallel:.2f}x")
    print("Shared runners are noisy; do not interpret as a guaranteed speedup.")


if __name__ == "__main__":
    main()
