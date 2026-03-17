"""
Newsvendor Problem (Single-Period Stochastic Inventory)

A retailer must decide how many units of a perishable product to order
before observing uncertain demand. Over-ordering incurs salvage loss;
under-ordering incurs lost profit.

Notation: NV | stochastic demand | min E[cost]

Complexity: O(1) for known distributions (closed-form critical fractile),
            O(S log S) for discrete scenario-based (sort + scan).

References:
    - Arrow, K.J., Harris, T. & Marschak, J. (1951). Optimal inventory policy.
      Econometrica, 19(3), 250-272. https://doi.org/10.2307/1906813
    - Silver, E.A., Pyke, D.F. & Thomas, D.J. (2017). Inventory and Production
      Management in Supply Chains, 4th ed. CRC Press.
"""
from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


@dataclass
class NewsvendorInstance:
    """Single-period newsvendor problem instance.

    Args:
        unit_cost: Cost per unit ordered (purchase price c).
        selling_price: Revenue per unit sold (p > c).
        salvage_value: Revenue per unsold unit (v < c).
        demand_scenarios: Array of demand values (discrete scenarios).
        probabilities: Probability of each scenario (sums to 1).
            If None, uniform probabilities assumed.
    """
    unit_cost: float
    selling_price: float
    salvage_value: float
    demand_scenarios: np.ndarray
    probabilities: np.ndarray | None = None

    def __post_init__(self):
        self.demand_scenarios = np.asarray(self.demand_scenarios, dtype=float)
        if self.probabilities is None:
            n = len(self.demand_scenarios)
            self.probabilities = np.full(n, 1.0 / n)
        else:
            self.probabilities = np.asarray(self.probabilities, dtype=float)

    @property
    def n_scenarios(self) -> int:
        return len(self.demand_scenarios)

    @property
    def overage_cost(self) -> float:
        """Cost of ordering one unit too many: c_o = c - v."""
        return self.unit_cost - self.salvage_value

    @property
    def underage_cost(self) -> float:
        """Cost of ordering one unit too few: c_u = p - c."""
        return self.selling_price - self.unit_cost

    @property
    def critical_fractile(self) -> float:
        """Optimal service level: c_u / (c_u + c_o)."""
        cu = self.underage_cost
        co = self.overage_cost
        return cu / (cu + co)

    def expected_cost(self, order_qty: float) -> float:
        """Compute expected cost for a given order quantity.

        Cost = c * Q + E[c_o * max(0, Q-D) + c_u * max(0, D-Q) - p*D + v*0]
        Simplified: E[cost] = c_o * E[max(0,Q-D)] + c_u * E[max(0,D-Q)]

        Args:
            order_qty: Number of units to order.

        Returns:
            Expected total mismatch cost (overage + underage).
        """
        co = self.overage_cost
        cu = self.underage_cost
        overage = np.maximum(0, order_qty - self.demand_scenarios)
        underage = np.maximum(0, self.demand_scenarios - order_qty)
        costs = co * overage + cu * underage
        return float(np.dot(costs, self.probabilities))

    def expected_profit(self, order_qty: float) -> float:
        """Compute expected profit for a given order quantity.

        Profit = p * E[min(Q, D)] + v * E[max(0, Q-D)] - c * Q

        Args:
            order_qty: Number of units to order.

        Returns:
            Expected profit.
        """
        sales = np.minimum(order_qty, self.demand_scenarios)
        leftover = np.maximum(0, order_qty - self.demand_scenarios)
        profits = (self.selling_price * sales
                   + self.salvage_value * leftover
                   - self.unit_cost * order_qty)
        return float(np.dot(profits, self.probabilities))

    @classmethod
    def random(cls, n_scenarios: int = 50, seed: int = 42) -> NewsvendorInstance:
        """Generate a random newsvendor instance.

        Args:
            n_scenarios: Number of demand scenarios.
            seed: Random seed.

        Returns:
            Random NewsvendorInstance.
        """
        rng = np.random.default_rng(seed)
        selling_price = rng.uniform(8, 20)
        unit_cost = rng.uniform(3, selling_price * 0.7)
        salvage_value = rng.uniform(0, unit_cost * 0.5)
        mean_demand = rng.uniform(50, 200)
        demand_scenarios = np.maximum(0, rng.normal(mean_demand, mean_demand * 0.3, n_scenarios))
        return cls(
            unit_cost=round(unit_cost, 2),
            selling_price=round(selling_price, 2),
            salvage_value=round(salvage_value, 2),
            demand_scenarios=demand_scenarios,
        )


@dataclass
class NewsvendorSolution:
    """Solution to the newsvendor problem.

    Args:
        order_quantity: Optimal order quantity Q*.
        expected_cost: Expected mismatch cost at Q*.
        expected_profit: Expected profit at Q*.
        service_level: P(D <= Q*), fill-rate probability.
    """
    order_quantity: float
    expected_cost: float
    expected_profit: float
    service_level: float

    def __repr__(self) -> str:
        return (f"NewsvendorSolution(Q*={self.order_quantity:.1f}, "
                f"E[cost]={self.expected_cost:.2f}, "
                f"E[profit]={self.expected_profit:.2f}, "
                f"SL={self.service_level:.3f})")


if __name__ == "__main__":
    inst = NewsvendorInstance(
        unit_cost=5.0,
        selling_price=10.0,
        salvage_value=2.0,
        demand_scenarios=np.array([40, 50, 60, 70, 80, 90, 100]),
    )
    print(f"Critical fractile: {inst.critical_fractile:.3f}")
    print(f"Overage cost: {inst.overage_cost}, Underage cost: {inst.underage_cost}")
    for q in [50, 60, 70, 80, 90]:
        print(f"  Q={q}: E[cost]={inst.expected_cost(q):.2f}, E[profit]={inst.expected_profit(q):.2f}")
