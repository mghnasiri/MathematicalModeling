"""
Capacitated Lot Sizing Problem (CLSP) — Instance and Solution definitions.

The CLSP extends the dynamic lot sizing problem by adding production
capacity constraints per period. The objective is to minimize total
fixed ordering costs, variable production costs, and holding costs
while satisfying all demands and respecting capacity limits.

Complexity: NP-hard (even for constant costs and capacities).

References:
    Florian, M., Lenstra, J.K. & Rinnooy Kan, A.H.G. (1980).
    Deterministic production planning: Algorithms and complexity.
    Management Science, 26(7), 669-679.
    https://doi.org/10.1287/mnsc.26.7.669

    Pochet, Y. & Wolsey, L.A. (2006). Production Planning by Mixed
    Integer Programming. Springer, New York.
    https://doi.org/10.1007/0-387-33477-7
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class CapLotSizingInstance:
    """Capacitated lot sizing problem instance.

    Attributes:
        T: Number of periods.
        demands: Demand per period, shape (T,).
        capacities: Production capacity per period, shape (T,).
        fixed_costs: Fixed ordering/setup cost per period, shape (T,).
        variable_costs: Variable production cost per unit per period, shape (T,).
        holding_costs: Holding cost per unit per period, shape (T,).
        name: Optional instance name.
    """

    T: int
    demands: np.ndarray
    capacities: np.ndarray
    fixed_costs: np.ndarray
    variable_costs: np.ndarray
    holding_costs: np.ndarray
    name: str = ""

    def __post_init__(self):
        self.demands = np.asarray(self.demands, dtype=float)
        self.capacities = np.asarray(self.capacities, dtype=float)
        self.fixed_costs = np.asarray(self.fixed_costs, dtype=float)
        self.variable_costs = np.asarray(self.variable_costs, dtype=float)
        self.holding_costs = np.asarray(self.holding_costs, dtype=float)

        for attr, name in [
            (self.demands, "demands"),
            (self.capacities, "capacities"),
            (self.fixed_costs, "fixed_costs"),
            (self.variable_costs, "variable_costs"),
            (self.holding_costs, "holding_costs"),
        ]:
            if attr.shape != (self.T,):
                raise ValueError(f"{name} shape {attr.shape} != ({self.T},)")

        if np.any(self.demands < 0):
            raise ValueError("demands must be non-negative")
        if np.any(self.capacities < 0):
            raise ValueError("capacities must be non-negative")

    @classmethod
    def random(
        cls,
        T: int = 10,
        demand_range: tuple[float, float] = (10.0, 80.0),
        capacity_factor: float = 1.5,
        fixed_cost_range: tuple[float, float] = (50.0, 200.0),
        variable_cost_range: tuple[float, float] = (1.0, 10.0),
        holding_cost_range: tuple[float, float] = (0.5, 5.0),
        seed: int | None = None,
    ) -> CapLotSizingInstance:
        """Generate a random capacitated lot sizing instance.

        Args:
            T: Number of periods.
            demand_range: Range for per-period demands.
            capacity_factor: Capacity as a multiple of average demand.
            fixed_cost_range: Range for fixed ordering costs.
            variable_cost_range: Range for variable costs per unit.
            holding_cost_range: Range for holding costs.
            seed: Random seed for reproducibility.

        Returns:
            A random CapLotSizingInstance.
        """
        rng = np.random.default_rng(seed)
        demands = np.round(rng.uniform(*demand_range, size=T)).astype(float)
        avg_demand = np.mean(demands)
        capacities = np.full(T, round(avg_demand * capacity_factor, 1))
        fixed_costs = np.round(rng.uniform(*fixed_cost_range, size=T)).astype(float)
        variable_costs = np.round(rng.uniform(*variable_cost_range, size=T), 2)
        holding_costs = np.round(rng.uniform(*holding_cost_range, size=T), 2)

        return cls(
            T=T,
            demands=demands,
            capacities=capacities,
            fixed_costs=fixed_costs,
            variable_costs=variable_costs,
            holding_costs=holding_costs,
            name=f"random_{T}",
        )

    def is_feasible(self, production: np.ndarray) -> bool:
        """Check if a production plan is feasible.

        Args:
            production: Production quantities per period, shape (T,).

        Returns:
            True if capacity and demand constraints are satisfied.
        """
        production = np.asarray(production, dtype=float)
        if np.any(production < -1e-10):
            return False
        if np.any(production > self.capacities + 1e-10):
            return False
        inventory = 0.0
        for t in range(self.T):
            inventory += production[t] - self.demands[t]
            if inventory < -1e-10:
                return False
        return True

    def compute_cost(self, production: np.ndarray) -> float:
        """Compute total cost of a production plan.

        Args:
            production: Production quantities per period, shape (T,).

        Returns:
            Total fixed + variable + holding cost.
        """
        production = np.asarray(production, dtype=float)
        total_cost = 0.0
        inventory = 0.0
        for t in range(self.T):
            if production[t] > 1e-10:
                total_cost += self.fixed_costs[t]
                total_cost += self.variable_costs[t] * production[t]
            inventory += production[t] - self.demands[t]
            if inventory < -1e-10:
                return float("inf")
            total_cost += self.holding_costs[t] * inventory
        return total_cost


@dataclass
class CapLotSizingSolution:
    """Solution to a capacitated lot sizing instance.

    Attributes:
        production: Production quantity per period, shape (T,).
        total_cost: Total fixed + variable + holding cost.
        production_periods: List of periods with positive production.
    """

    production: np.ndarray
    total_cost: float
    production_periods: list[int]

    def __repr__(self) -> str:
        return (
            f"CapLotSizingSolution(cost={self.total_cost:.2f}, "
            f"produce_in={self.production_periods})"
        )


# ── Benchmark instances ──────────────────────────────────────────────────────


def tight_capacity_6() -> CapLotSizingInstance:
    """6-period instance with tight capacity.

    Capacity barely exceeds demand, forcing production in most periods.
    """
    return CapLotSizingInstance(
        T=6,
        demands=np.array([20.0, 30.0, 25.0, 35.0, 20.0, 30.0]),
        capacities=np.array([40.0, 40.0, 40.0, 40.0, 40.0, 40.0]),
        fixed_costs=np.full(6, 100.0),
        variable_costs=np.full(6, 2.0),
        holding_costs=np.full(6, 1.0),
        name="tight_6",
    )


def loose_capacity_4() -> CapLotSizingInstance:
    """4-period instance with loose capacity (similar to uncapacitated)."""
    return CapLotSizingInstance(
        T=4,
        demands=np.array([20.0, 50.0, 10.0, 50.0]),
        capacities=np.array([200.0, 200.0, 200.0, 200.0]),
        fixed_costs=np.full(4, 54.0),
        variable_costs=np.full(4, 0.0),
        holding_costs=np.full(4, 1.0),
        name="loose_4",
    )


def variable_costs_8() -> CapLotSizingInstance:
    """8-period instance with varying costs and capacities."""
    return CapLotSizingInstance(
        T=8,
        demands=np.array([15.0, 25.0, 35.0, 20.0, 30.0, 10.0, 25.0, 20.0]),
        capacities=np.array([50.0, 50.0, 60.0, 45.0, 55.0, 40.0, 50.0, 50.0]),
        fixed_costs=np.array([80.0, 90.0, 70.0, 100.0, 85.0, 75.0, 95.0, 80.0]),
        variable_costs=np.array([3.0, 2.5, 4.0, 3.5, 2.0, 3.0, 2.5, 3.5]),
        holding_costs=np.full(8, 1.5),
        name="variable_8",
    )


if __name__ == "__main__":
    inst = tight_capacity_6()
    print(f"{inst.name}: T={inst.T}")
    print(f"  demands: {inst.demands}")
    print(f"  capacities: {inst.capacities}")
    print(f"  total demand: {np.sum(inst.demands)}")
    print(f"  total capacity: {np.sum(inst.capacities)}")
