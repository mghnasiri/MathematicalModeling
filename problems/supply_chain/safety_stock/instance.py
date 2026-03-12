"""
Safety Stock Optimization — Instance and Solution definitions.

Problem: Determine optimal safety stock levels for items with stochastic
demand and/or lead times, balancing holding cost against stockout risk.

Uses standard analytical formulas (normal distribution approximation):
- Safety stock = z * sigma_DDLT
- sigma_DDLT = sqrt(L * sigma_D^2 + D^2 * sigma_L^2)

where z = Phi^{-1}(service_level), L = lead time, D = demand rate,
sigma_D = demand std, sigma_L = lead time std.

References:
    Silver, E.A., Pyke, D.F. & Thomas, D.J. (2016). Inventory and
    Production Management in Supply Chains. 4th edition, CRC Press.

    Chopra, S. & Meindl, P. (2019). Supply Chain Management: Strategy,
    Planning, and Operation. 7th edition, Pearson.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class SafetyStockInstance:
    """Safety stock optimization instance for multiple items.

    Attributes:
        n: Number of items (SKUs).
        mean_demands: Mean demand per period for each item, shape (n,).
        std_demands: Standard deviation of demand per period, shape (n,).
        mean_lead_times: Mean lead time (periods) for each item, shape (n,).
        std_lead_times: Std dev of lead time for each item, shape (n,).
        holding_costs: Holding cost per unit per period, shape (n,).
        service_level: Target service level (cycle service level).
        name: Optional instance name.
    """

    n: int
    mean_demands: np.ndarray
    std_demands: np.ndarray
    mean_lead_times: np.ndarray
    std_lead_times: np.ndarray
    holding_costs: np.ndarray
    service_level: float = 0.95
    name: str = ""

    def __post_init__(self):
        self.mean_demands = np.asarray(self.mean_demands, dtype=float)
        self.std_demands = np.asarray(self.std_demands, dtype=float)
        self.mean_lead_times = np.asarray(self.mean_lead_times, dtype=float)
        self.std_lead_times = np.asarray(self.std_lead_times, dtype=float)
        self.holding_costs = np.asarray(self.holding_costs, dtype=float)

        for attr, attr_name in [
            (self.mean_demands, "mean_demands"),
            (self.std_demands, "std_demands"),
            (self.mean_lead_times, "mean_lead_times"),
            (self.std_lead_times, "std_lead_times"),
            (self.holding_costs, "holding_costs"),
        ]:
            if attr.shape != (self.n,):
                raise ValueError(f"{attr_name} shape {attr.shape} != ({self.n},)")

        if np.any(self.mean_demands < 0):
            raise ValueError("mean_demands must be non-negative")
        if np.any(self.std_demands < 0):
            raise ValueError("std_demands must be non-negative")
        if np.any(self.mean_lead_times < 0):
            raise ValueError("mean_lead_times must be non-negative")
        if not (0 < self.service_level < 1):
            raise ValueError("service_level must be in (0, 1)")

    @classmethod
    def random(
        cls,
        n: int = 5,
        seed: int | None = None,
    ) -> SafetyStockInstance:
        """Generate a random safety stock instance.

        Args:
            n: Number of items.
            seed: Random seed.

        Returns:
            A random SafetyStockInstance.
        """
        rng = np.random.default_rng(seed)
        mean_demands = np.round(rng.uniform(10.0, 200.0, size=n), 1)
        std_demands = np.round(mean_demands * rng.uniform(0.1, 0.5, size=n), 1)
        mean_lead_times = np.round(rng.uniform(1.0, 8.0, size=n), 1)
        std_lead_times = np.round(rng.uniform(0.0, 2.0, size=n), 1)
        holding_costs = np.round(rng.uniform(0.5, 10.0, size=n), 2)

        return cls(
            n=n,
            mean_demands=mean_demands,
            std_demands=std_demands,
            mean_lead_times=mean_lead_times,
            std_lead_times=std_lead_times,
            holding_costs=holding_costs,
            name=f"random_{n}",
        )


@dataclass
class SafetyStockSolution:
    """Solution to a safety stock optimization instance.

    Attributes:
        safety_stocks: Safety stock for each item, shape (n,).
        reorder_points: Reorder point for each item, shape (n,).
        total_holding_cost: Total holding cost per period.
    """

    safety_stocks: np.ndarray
    reorder_points: np.ndarray
    total_holding_cost: float

    def __repr__(self) -> str:
        return (
            f"SafetyStockSolution(cost={self.total_holding_cost:.2f}, "
            f"ss={self.safety_stocks.tolist()})"
        )


# -- Benchmark instances ------------------------------------------------------


def basic_3items() -> SafetyStockInstance:
    """3 items with different demand variability."""
    return SafetyStockInstance(
        n=3,
        mean_demands=np.array([100.0, 50.0, 200.0]),
        std_demands=np.array([20.0, 15.0, 40.0]),
        mean_lead_times=np.array([2.0, 3.0, 1.0]),
        std_lead_times=np.array([0.5, 1.0, 0.0]),
        holding_costs=np.array([5.0, 8.0, 3.0]),
        service_level=0.95,
        name="basic_3",
    )


def single_item() -> SafetyStockInstance:
    """Single item for validation."""
    return SafetyStockInstance(
        n=1,
        mean_demands=np.array([100.0]),
        std_demands=np.array([25.0]),
        mean_lead_times=np.array([4.0]),
        std_lead_times=np.array([1.0]),
        holding_costs=np.array([2.0]),
        service_level=0.95,
        name="single_1",
    )


def zero_lt_variability() -> SafetyStockInstance:
    """Items with deterministic (zero-variance) lead times."""
    return SafetyStockInstance(
        n=2,
        mean_demands=np.array([80.0, 120.0]),
        std_demands=np.array([16.0, 30.0]),
        mean_lead_times=np.array([3.0, 2.0]),
        std_lead_times=np.array([0.0, 0.0]),
        holding_costs=np.array([4.0, 6.0]),
        service_level=0.90,
        name="zero_lt_var_2",
    )


if __name__ == "__main__":
    inst = basic_3items()
    print(f"{inst.name}: n={inst.n}, SL={inst.service_level}")
    for i in range(inst.n):
        print(f"  Item {i}: D={inst.mean_demands[i]}, sigma_D={inst.std_demands[i]}, "
              f"LT={inst.mean_lead_times[i]}, sigma_LT={inst.std_lead_times[i]}")
