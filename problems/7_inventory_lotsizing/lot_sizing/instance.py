"""
Dynamic Lot Sizing — Instance and Solution definitions.

The dynamic lot sizing problem determines when and how much to order
over a finite planning horizon of T periods with time-varying demands,
to minimize total ordering plus holding costs.

Complexity: O(T^2) via Wagner-Whitin dynamic programming.

References:
    Wagner, H.M. & Whitin, T.M. (1958). Dynamic version of the economic
    lot size model. Management Science, 5(1), 89-96.
    https://doi.org/10.1287/mnsc.5.1.89

    Silver, E.A. & Meal, H.C. (1973). A heuristic for selecting lot size
    quantities for the case of a deterministic time-varying demand rate
    and discrete opportunities for replenishment. Production and Inventory
    Management, 14(2), 64-74.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class LotSizingInstance:
    """Dynamic lot sizing problem instance.

    Attributes:
        T: Number of periods.
        demands: Demand per period, shape (T,).
        ordering_costs: Fixed ordering cost per period, shape (T,).
        holding_costs: Holding cost per unit per period, shape (T,).
        name: Optional instance name.
    """

    T: int
    demands: np.ndarray
    ordering_costs: np.ndarray
    holding_costs: np.ndarray
    name: str = ""

    def __post_init__(self):
        self.demands = np.asarray(self.demands, dtype=float)
        self.ordering_costs = np.asarray(self.ordering_costs, dtype=float)
        self.holding_costs = np.asarray(self.holding_costs, dtype=float)

        if self.demands.shape != (self.T,):
            raise ValueError(
                f"demands shape {self.demands.shape} != ({self.T},)"
            )
        if self.ordering_costs.shape != (self.T,):
            raise ValueError(
                f"ordering_costs shape {self.ordering_costs.shape} != ({self.T},)"
            )
        if self.holding_costs.shape != (self.T,):
            raise ValueError(
                f"holding_costs shape {self.holding_costs.shape} != ({self.T},)"
            )
        if np.any(self.demands < 0):
            raise ValueError("demands must be non-negative")

    @classmethod
    def random(
        cls,
        T: int = 12,
        demand_range: tuple[float, float] = (10.0, 100.0),
        ordering_cost_range: tuple[float, float] = (50.0, 200.0),
        holding_cost_range: tuple[float, float] = (0.5, 5.0),
        seed: int | None = None,
    ) -> LotSizingInstance:
        """Generate a random lot sizing instance.

        Args:
            T: Number of periods.
            demand_range: Range for per-period demands.
            ordering_cost_range: Range for ordering costs.
            holding_cost_range: Range for holding costs.
            seed: Random seed for reproducibility.

        Returns:
            A random LotSizingInstance.
        """
        rng = np.random.default_rng(seed)
        demands = np.round(rng.uniform(*demand_range, size=T)).astype(float)
        ordering = np.round(rng.uniform(*ordering_cost_range, size=T)).astype(float)
        holding = np.round(rng.uniform(*holding_cost_range, size=T), 2)

        return cls(
            T=T,
            demands=demands,
            ordering_costs=ordering,
            holding_costs=holding,
            name=f"random_{T}",
        )

    def compute_cost(self, order_quantities: np.ndarray) -> float:
        """Compute total cost for given order quantities.

        Args:
            order_quantities: Order amount per period, shape (T,).

        Returns:
            Total ordering + holding cost.
        """
        order_quantities = np.asarray(order_quantities, dtype=float)
        total_cost = 0.0
        inventory = 0.0
        for t in range(self.T):
            if order_quantities[t] > 0:
                total_cost += self.ordering_costs[t]
            inventory += order_quantities[t] - self.demands[t]
            if inventory < -1e-10:
                return float("inf")  # demand not met
            total_cost += self.holding_costs[t] * inventory
        return total_cost


@dataclass
class LotSizingSolution:
    """Solution to a dynamic lot sizing instance.

    Attributes:
        order_quantities: Order amount per period, shape (T,).
        total_cost: Total ordering + holding cost.
        order_periods: List of periods where an order is placed.
    """

    order_quantities: np.ndarray
    total_cost: float
    order_periods: list[int]

    def __repr__(self) -> str:
        return (
            f"LotSizingSolution(cost={self.total_cost:.2f}, "
            f"orders_in={self.order_periods})"
        )


# ── Benchmark instances ──────────────────────────────────────────────────────


def textbook_4period() -> LotSizingInstance:
    """4-period textbook instance.

    Demands: [20, 50, 10, 50], K=54, h=1 per period.
    Wagner-Whitin optimal: order in periods 0 and 1 (or 0 and 3).
    """
    return LotSizingInstance(
        T=4,
        demands=np.array([20.0, 50.0, 10.0, 50.0]),
        ordering_costs=np.array([54.0, 54.0, 54.0, 54.0]),
        holding_costs=np.array([1.0, 1.0, 1.0, 1.0]),
        name="textbook_4",
    )


def seasonal_8period() -> LotSizingInstance:
    """8-period instance with seasonal demand pattern."""
    return LotSizingInstance(
        T=8,
        demands=np.array([10.0, 20.0, 30.0, 40.0, 40.0, 30.0, 20.0, 10.0]),
        ordering_costs=np.full(8, 100.0),
        holding_costs=np.full(8, 2.0),
        name="seasonal_8",
    )


def varying_costs_6period() -> LotSizingInstance:
    """6-period instance with varying ordering and holding costs."""
    return LotSizingInstance(
        T=6,
        demands=np.array([30.0, 10.0, 45.0, 25.0, 15.0, 35.0]),
        ordering_costs=np.array([80.0, 90.0, 70.0, 85.0, 75.0, 95.0]),
        holding_costs=np.array([1.5, 2.0, 1.0, 1.5, 2.5, 1.0]),
        name="varying_6",
    )


if __name__ == "__main__":
    inst = textbook_4period()
    print(f"{inst.name}: T={inst.T}")
    print(f"  demands: {inst.demands}")
    print(f"  K: {inst.ordering_costs}")
    print(f"  h: {inst.holding_costs}")
