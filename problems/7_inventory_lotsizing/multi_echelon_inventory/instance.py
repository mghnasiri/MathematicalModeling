"""
Multi-Echelon Inventory — Instance and Solution definitions.

A serial supply chain with L echelons (stages), where each echelon holds
inventory to buffer against demand variability. Material flows from
echelon L (most upstream, e.g. raw material) to echelon 1 (most downstream,
facing customer demand).

The goal is to determine base-stock levels at each echelon to minimize
total expected holding cost while achieving a target service level.

Complexity: Finding optimal base-stock levels is generally hard for
general networks; serial systems admit efficient solutions via Clark-Scarf
decomposition.

References:
    Clark, A.J. & Scarf, H. (1960). Optimal policies for a multi-echelon
    inventory problem. Management Science, 6(4), 475-490.
    https://doi.org/10.1287/mnsc.6.4.475

    Axsater, S. (2006). Inventory Control. 2nd edition, Springer.
    https://doi.org/10.1007/0-387-33331-2

    Roundy, R. (1985). 98%-effective integer-ratio lot-sizing for
    one-warehouse multi-retailer systems. Management Science, 31(11),
    1416-1430. https://doi.org/10.1287/mnsc.31.11.1416
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class MultiEchelonInstance:
    """Multi-echelon serial inventory problem instance.

    Echelons are numbered 1..L, where echelon 1 faces the customer.
    Material flows from echelon L to echelon 1.

    Attributes:
        L: Number of echelons (stages).
        holding_costs: Echelon holding cost per unit per period, shape (L,).
            holding_costs[i] is for echelon i+1 (0-indexed).
        ordering_costs: Fixed ordering cost per echelon, shape (L,).
        lead_times: Lead time (periods) for each echelon, shape (L,).
        mean_demand: Mean customer demand per period.
        std_demand: Standard deviation of customer demand per period.
        service_level: Target fill rate / service level (e.g. 0.95).
        name: Optional instance name.
    """

    L: int
    holding_costs: np.ndarray
    ordering_costs: np.ndarray
    lead_times: np.ndarray
    mean_demand: float
    std_demand: float
    service_level: float = 0.95
    name: str = ""

    def __post_init__(self):
        self.holding_costs = np.asarray(self.holding_costs, dtype=float)
        self.ordering_costs = np.asarray(self.ordering_costs, dtype=float)
        self.lead_times = np.asarray(self.lead_times, dtype=float)

        if self.holding_costs.shape != (self.L,):
            raise ValueError(
                f"holding_costs shape {self.holding_costs.shape} != ({self.L},)"
            )
        if self.ordering_costs.shape != (self.L,):
            raise ValueError(
                f"ordering_costs shape {self.ordering_costs.shape} != ({self.L},)"
            )
        if self.lead_times.shape != (self.L,):
            raise ValueError(
                f"lead_times shape {self.lead_times.shape} != ({self.L},)"
            )
        if self.mean_demand <= 0:
            raise ValueError("mean_demand must be positive")
        if self.std_demand < 0:
            raise ValueError("std_demand must be non-negative")
        if not (0 < self.service_level < 1):
            raise ValueError("service_level must be in (0, 1)")

    @classmethod
    def random(
        cls,
        L: int = 3,
        mean_demand: float = 100.0,
        std_demand: float = 20.0,
        seed: int | None = None,
    ) -> MultiEchelonInstance:
        """Generate a random multi-echelon instance.

        Holding costs decrease upstream (cheaper to hold raw materials).
        Lead times are random integers 1-4.

        Args:
            L: Number of echelons.
            mean_demand: Mean customer demand.
            std_demand: Std deviation of demand.
            seed: Random seed for reproducibility.

        Returns:
            A random MultiEchelonInstance.
        """
        rng = np.random.default_rng(seed)

        # Holding costs decrease upstream
        base_h = rng.uniform(5.0, 15.0)
        holding_costs = np.array([
            base_h * (1.0 - 0.2 * i) for i in range(L)
        ])
        holding_costs = np.maximum(holding_costs, 0.5)

        ordering_costs = np.round(rng.uniform(50.0, 200.0, size=L)).astype(float)
        lead_times = rng.integers(1, 5, size=L).astype(float)

        return cls(
            L=L,
            holding_costs=np.round(holding_costs, 2),
            ordering_costs=ordering_costs,
            lead_times=lead_times,
            mean_demand=mean_demand,
            std_demand=std_demand,
            name=f"random_{L}",
        )

    def echelon_lead_time(self, echelon: int) -> float:
        """Cumulative lead time from echelon to customer (echelons 0..echelon).

        Args:
            echelon: Echelon index (0-indexed, 0 = closest to customer).

        Returns:
            Cumulative lead time.
        """
        return float(np.sum(self.lead_times[:echelon + 1]))

    def demand_during_lead_time(self, echelon: int) -> tuple[float, float]:
        """Mean and std of demand during echelon lead time.

        Args:
            echelon: Echelon index (0-indexed).

        Returns:
            (mean, std) of demand during cumulative lead time.
        """
        lt = self.echelon_lead_time(echelon)
        mean = self.mean_demand * lt
        std = self.std_demand * np.sqrt(lt)
        return float(mean), float(std)


@dataclass
class MultiEchelonSolution:
    """Solution to a multi-echelon inventory instance.

    Attributes:
        base_stock_levels: Base-stock level for each echelon, shape (L,).
        safety_stocks: Safety stock for each echelon, shape (L,).
        total_holding_cost: Expected total holding cost per period.
        total_cost: Total expected cost (holding + ordering) per period.
    """

    base_stock_levels: np.ndarray
    safety_stocks: np.ndarray
    total_holding_cost: float
    total_cost: float

    def __repr__(self) -> str:
        return (
            f"MultiEchelonSolution(cost={self.total_cost:.2f}, "
            f"holding={self.total_holding_cost:.2f}, "
            f"base_stocks={self.base_stock_levels.tolist()})"
        )


# ── Benchmark instances ──────────────────────────────────────────────────────


def serial_2echelon() -> MultiEchelonInstance:
    """Simple 2-echelon serial system.

    Echelon 1 (retail): h=10, K=100, LT=1
    Echelon 2 (warehouse): h=5, K=150, LT=2
    """
    return MultiEchelonInstance(
        L=2,
        holding_costs=np.array([10.0, 5.0]),
        ordering_costs=np.array([100.0, 150.0]),
        lead_times=np.array([1.0, 2.0]),
        mean_demand=50.0,
        std_demand=10.0,
        service_level=0.95,
        name="serial_2",
    )


def serial_3echelon() -> MultiEchelonInstance:
    """3-echelon serial system (retail-warehouse-factory).

    Echelon 1 (retail): h=12, K=80, LT=1
    Echelon 2 (warehouse): h=6, K=120, LT=2
    Echelon 3 (factory): h=2, K=200, LT=3
    """
    return MultiEchelonInstance(
        L=3,
        holding_costs=np.array([12.0, 6.0, 2.0]),
        ordering_costs=np.array([80.0, 120.0, 200.0]),
        lead_times=np.array([1.0, 2.0, 3.0]),
        mean_demand=100.0,
        std_demand=25.0,
        service_level=0.95,
        name="serial_3",
    )


def high_variance_3echelon() -> MultiEchelonInstance:
    """3-echelon system with high demand variability."""
    return MultiEchelonInstance(
        L=3,
        holding_costs=np.array([15.0, 8.0, 3.0]),
        ordering_costs=np.array([100.0, 150.0, 250.0]),
        lead_times=np.array([2.0, 3.0, 4.0]),
        mean_demand=200.0,
        std_demand=80.0,
        service_level=0.98,
        name="high_var_3",
    )


if __name__ == "__main__":
    inst = serial_3echelon()
    print(f"{inst.name}: L={inst.L}, D~N({inst.mean_demand}, {inst.std_demand}^2)")
    for i in range(inst.L):
        lt = inst.echelon_lead_time(i)
        mu, sigma = inst.demand_during_lead_time(i)
        print(f"  Echelon {i+1}: h={inst.holding_costs[i]}, "
              f"K={inst.ordering_costs[i]}, LT={inst.lead_times[i]}, "
              f"cum_LT={lt}, DLT~N({mu:.1f}, {sigma:.1f}^2)")
