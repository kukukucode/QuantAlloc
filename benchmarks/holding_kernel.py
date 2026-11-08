"""Compare the reference pandas path and optional native kernel.

Run:
    python -m benchmarks.holding_kernel
    QUANTALLOC_ENGINE=rust python -m benchmarks.holding_kernel

Report timings without asserting a speedup (shared CI runners are noisy).
"""

import importlib.util
import statistics
import time

import numpy as np
import pandas as pd

from src.fast_backend import _python_holding, simulate_holding_period


def measure(fn, repetitions=9):
    samples = []
    for _ in range(repetitions):
        started = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def main():
    rng = np.random.default_rng(14)
    native_available = importlib.util.find_spec("_quantalloc_rust") is not None
    print(f"Rust extension installed: {native_available}")
    for days, assets in [(63, 10), (252, 30), (252, 300), (252, 1000)]:
        values = rng.normal(0.0, 0.01, size=(days, assets))
        returns = pd.DataFrame(values)
        weights = pd.Series(np.full(assets, 1.0 / assets))
        reference = _python_holding(weights, returns)
        python_seconds = measure(lambda: _python_holding(weights, returns))
        if not native_available:
            print(
                f"{days:>4} x {assets:>4} | pandas={python_seconds:.6f}s"
                " | native=not installed"
            )
            continue
        import os

        old_mode = os.environ.get("QUANTALLOC_ENGINE")
        try:
            os.environ["QUANTALLOC_ENGINE"] = "rust"
            comparison = simulate_holding_period(weights, returns)
            np.testing.assert_allclose(
                comparison[0], reference[0], atol=1e-12, rtol=1e-10
            )
            np.testing.assert_allclose(
                comparison[1], reference[1], atol=1e-12, rtol=1e-10
            )
            native_seconds = measure(
                lambda: simulate_holding_period(weights, returns)
            )
        finally:
            if old_mode is None:
                os.environ.pop("QUANTALLOC_ENGINE", None)
            else:
                os.environ["QUANTALLOC_ENGINE"] = old_mode
        print(
            f"{days:>4} x {assets:>4} | pandas={python_seconds:.6f}s"
            f" | Rust={native_seconds:.6f}s"
            f" | ratio={python_seconds / native_seconds:.2f}x"
        )


if __name__ == "__main__":
    main()
