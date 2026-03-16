"""
Agricultural Feed & Fertilizer Procurement Planning Problem

Domain: Dairy farming / Agricultural input procurement
Notation: LS_agri | seasonal demand, multiple inputs | min total cost

A dairy farm plans procurement of agricultural inputs (cattle feed,
fertilizer, seeds) over a 12-month horizon. Each input has seasonal
demand patterns and specific ordering/holding costs. The problem
compares EOQ (steady-state), Silver-Meal (dynamic heuristic), and
Wagner-Whitin (optimal DP) lot sizing strategies.

Complexity:
    - EOQ: O(1) closed-form
    - Silver-Meal: O(T^2) per input
    - Wagner-Whitin: O(T^2) per input, optimal

References:
    Wagner, H.M. & Whitin, T.M. (1958). Dynamic version of the economic
    lot size model. Management Science, 5(1), 89-96.
    https://doi.org/10.1287/mnsc.5.1.89

    Silver, E.A. & Meal, H.C. (1973). A heuristic for selecting lot size
    quantities for the case of a deterministic time-varying demand rate
    and discrete opportunities for replenishment. Production and Inventory
    Management, 14(2), 64-74.

    Boyabatli, O., Nasiry, J. & Zhou, Y.H. (2019). Crop planning in
    sustainable agriculture: Dynamic farmland allocation in the presence
    of crop rotation benefits. Management Science, 65(5), 2060-2076.
    https://doi.org/10.1287/mnsc.2018.3044
"""
from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

MONTHS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


@dataclass
class FarmInputProfile:
    """Profile for a single agricultural input.

    Args:
        name: Input name (e.g., "Cattle Feed").
        unit: Measurement unit (e.g., "tons").
        ordering_cost: Fixed cost per order ($).
        holding_cost_per_unit_per_month: Holding cost ($/unit/month).
        monthly_demands: Array of 12 monthly demand values.
    """
    name: str
    unit: str
    ordering_cost: float
    holding_cost_per_unit_per_month: float
    monthly_demands: np.ndarray

    def __post_init__(self):
        self.monthly_demands = np.asarray(self.monthly_demands, dtype=float)

    @property
    def total_annual_demand(self) -> float:
        return float(np.sum(self.monthly_demands))

    @property
    def avg_monthly_demand(self) -> float:
        nonzero = self.monthly_demands[self.monthly_demands > 0]
        return float(np.mean(nonzero)) if len(nonzero) > 0 else 0.0

    @property
    def n_active_months(self) -> int:
        return int(np.sum(self.monthly_demands > 0))


@dataclass
class FeedProcurementInstance:
    """Multi-input agricultural procurement planning instance.

    Args:
        inputs: List of FarmInputProfile for each input.
        horizon: Number of periods (months).
        name: Instance name.
    """
    inputs: list[FarmInputProfile]
    horizon: int = 12
    name: str = "feed_procurement"

    @property
    def n_inputs(self) -> int:
        return len(self.inputs)

    @classmethod
    def quebec_dairy_farm(cls) -> FeedProcurementInstance:
        """Create the Quebec dairy farm benchmark instance.

        500-head dairy farm with three agricultural inputs:
        - Cattle feed: steady demand with winter peak
        - Fertilizer: spring/summer only
        - Seeds: spring planting window only

        Returns:
            FeedProcurementInstance with 3 inputs.
        """
        inputs = [
            FarmInputProfile(
                name="Cattle Feed (grain mix)",
                unit="tons",
                ordering_cost=150.0,
                holding_cost_per_unit_per_month=2.0,
                monthly_demands=np.array([
                    75.0, 72.0, 65.0, 55.0, 45.0, 40.0,
                    42.0, 45.0, 50.0, 58.0, 70.0, 80.0,
                ]),
            ),
            FarmInputProfile(
                name="Fertilizer (NPK 15-15-15)",
                unit="tons",
                ordering_cost=200.0,
                holding_cost_per_unit_per_month=5.0,
                monthly_demands=np.array([
                    0.0, 0.0, 15.0, 45.0, 60.0, 40.0,
                    25.0, 10.0, 0.0, 0.0, 0.0, 0.0,
                ]),
            ),
            FarmInputProfile(
                name="Seeds (hay/silage mix)",
                unit="tons",
                ordering_cost=100.0,
                holding_cost_per_unit_per_month=3.0,
                monthly_demands=np.array([
                    0.0, 0.0, 8.0, 15.0, 12.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ]),
            ),
        ]
        return cls(inputs=inputs, horizon=12, name="quebec_dairy_farm")

    @classmethod
    def random(cls, n_inputs: int = 3, seed: int = 42) -> FeedProcurementInstance:
        """Generate a random procurement instance.

        Args:
            n_inputs: Number of agricultural inputs.
            seed: Random seed.

        Returns:
            Random FeedProcurementInstance.
        """
        rng = np.random.default_rng(seed)
        names = ["Input_A", "Input_B", "Input_C", "Input_D", "Input_E"]
        inputs = []
        for i in range(n_inputs):
            base_demand = rng.uniform(10, 80)
            seasonal = rng.uniform(0.5, 1.5, 12)
            demands = np.maximum(0, base_demand * seasonal)
            # Possibly zero-out some months for seasonal inputs
            if rng.random() > 0.5:
                zero_months = rng.choice(12, size=rng.integers(3, 7), replace=False)
                demands[zero_months] = 0.0
            inputs.append(FarmInputProfile(
                name=names[i % len(names)],
                unit="tons",
                ordering_cost=rng.uniform(50, 300),
                holding_cost_per_unit_per_month=rng.uniform(1, 10),
                monthly_demands=np.round(demands, 1),
            ))
        return cls(inputs=inputs, horizon=12, name=f"random_{n_inputs}inputs")


@dataclass
class InputProcurementSolution:
    """Solution for a single input's procurement plan.

    Args:
        input_name: Name of the agricultural input.
        method: Algorithm used.
        total_cost: Total ordering + holding cost.
        order_quantities: Array of order quantities per period.
        order_periods: List of period indices where orders are placed.
        num_orders: Number of orders placed.
    """
    input_name: str
    method: str
    total_cost: float
    order_quantities: np.ndarray | list[float]
    order_periods: list[int]
    num_orders: int

    def __repr__(self) -> str:
        return (f"InputProcurementSolution({self.input_name}, "
                f"method={self.method}, cost=${self.total_cost:.2f}, "
                f"orders={self.num_orders})")


@dataclass
class FeedProcurementSolution:
    """Aggregate solution for all agricultural inputs.

    Args:
        input_solutions: Dict mapping input key to method results.
        total_cost: Sum of all input costs.
        method: Algorithm used.
    """
    input_solutions: dict[str, InputProcurementSolution]
    total_cost: float
    method: str

    def __repr__(self) -> str:
        return (f"FeedProcurementSolution(method={self.method}, "
                f"total_cost=${self.total_cost:.2f}, "
                f"n_inputs={len(self.input_solutions)})")


if __name__ == "__main__":
    inst = FeedProcurementInstance.quebec_dairy_farm()
    print(f"Quebec dairy farm: {inst.n_inputs} inputs, {inst.horizon} months")
    for inp in inst.inputs:
        print(f"  {inp.name}: total={inp.total_annual_demand:.0f} {inp.unit}, "
              f"active months={inp.n_active_months}")
        print(f"    Demands: {inp.monthly_demands}")
