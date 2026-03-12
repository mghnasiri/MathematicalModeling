"""
Wagner-Whitin Lot Sizing Problem — Instance and Solution definitions.

Problem notation: Uncapacitated Lot Sizing (ULS)

Given a planning horizon of T periods with known demands d_t, fixed ordering
cost K_t, and holding cost h_t per unit per period, determine order quantities
to satisfy all demand at minimum total cost.

The Wagner-Whitin property guarantees an optimal solution where orders only
occur when inventory is zero (Zero Inventory Ordering).

Complexity: O(T^2) via dynamic programming.

References:
    Wagner, H.M. & Whitin, T.M. (1958). Dynamic version of the economic
    lot size model. Management Science, 5(1), 89-96.
    https://doi.org/10.1287/mnsc.5.1.89
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class WagnerWhitinInstance:
    """Wagner-Whitin lot sizing instance.

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

        for attr, attr_name in [
            (self.demands, "demands"),
            (self.ordering_costs, "ordering_costs"),
            (self.holding_costs, "holding_costs"),
        ]:
            if attr.shape != (self.T,):
                raise ValueError(f"{attr_name} shape {attr.shape} != ({self.T},)")

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
    ) -> WagnerWhitinInstance:
        """Generate a random Wagner-Whitin instance.

        Args:
            T: Number of periods.
            demand_range: Range for per-period demands.
            ordering_cost_range: Range for ordering costs.
            holding_cost_range: Range for holding costs.
            seed: Random seed for reproducibility.

        Returns:
            A random WagnerWhitinInstance.
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
            if order_quantities[t] > 1e-10:
                total_cost += self.ordering_costs[t]
            inventory += order_quantities[t] - self.demands[t]
            if inventory < -1e-10:
                return float("inf")
            total_cost += self.holding_costs[t] * inventory
        return total_cost


@dataclass
class WagnerWhitinSolution:
    """Solution to a Wagner-Whitin instance.

    Attributes:
        order_quantities: Order amount per period, shape (T,).
        total_cost: Total ordering + holding cost.
        order_periods: List of periods where an order is placed (0-indexed).
    """

    order_quantities: np.ndarray
    total_cost: float
    order_periods: list[int]

    def __repr__(self) -> str:
        return (
            f"WagnerWhitinSolution(cost={self.total_cost:.2f}, "
            f"orders_in={self.order_periods})"
        )


# -- Benchmark instances ------------------------------------------------------


def textbook_4() -> WagnerWhitinInstance:
    """Classic 4-period textbook instance.

    Demands: [20, 50, 10, 50], K=54, h=1.
    """
    return WagnerWhitinInstance(
        T=4,
        demands=np.array([20.0, 50.0, 10.0, 50.0]),
        ordering_costs=np.full(4, 54.0),
        holding_costs=np.full(4, 1.0),
        name="textbook_4",
    )


def seasonal_8() -> WagnerWhitinInstance:
    """8-period instance with seasonal demand pattern."""
    return WagnerWhitinInstance(
        T=8,
        demands=np.array([10.0, 20.0, 30.0, 40.0, 40.0, 30.0, 20.0, 10.0]),
        ordering_costs=np.full(8, 100.0),
        holding_costs=np.full(8, 2.0),
        name="seasonal_8",
    )


def single_period() -> WagnerWhitinInstance:
    """Trivial single-period instance."""
    return WagnerWhitinInstance(
        T=1,
        demands=np.array([30.0]),
        ordering_costs=np.array([50.0]),
        holding_costs=np.array([1.0]),
        name="single_1",
    )


if __name__ == "__main__":
    inst = textbook_4()
    print(f"{inst.name}: T={inst.T}, demands={inst.demands}")
