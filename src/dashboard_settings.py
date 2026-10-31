"""Validated settings shared by dashboard analyses."""

import math
from dataclasses import dataclass

from src.sample_config import (
    ESTIMATION_WINDOW,
    HOLDING_PERIOD,
    TRANSACTION_COST_RATE,
)


@dataclass(frozen=True)
class DashboardSettings:
    estimation_window: int = ESTIMATION_WINDOW
    holding_period: int = HOLDING_PERIOD
    transaction_cost_rate: float = TRANSACTION_COST_RATE

    def __post_init__(self) -> None:
        if (
            isinstance(self.estimation_window, bool)
            or not isinstance(self.estimation_window, int)
            or self.estimation_window < 2
        ):
            raise ValueError("invalid estimation_window")

        if (
            isinstance(self.holding_period, bool)
            or not isinstance(self.holding_period, int)
            or self.holding_period < 1
        ):
            raise ValueError("invalid holding_period")

        cost = self.transaction_cost_rate
        if (
            isinstance(cost, bool)
            or not isinstance(cost, (int, float))
            or not math.isfinite(cost)
            or not 0 <= cost < 1
        ):
            raise ValueError("invalid transaction_cost_rate")
