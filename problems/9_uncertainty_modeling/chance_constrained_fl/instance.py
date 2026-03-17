"""
Chance-Constrained Facility Location Problem

Extends the Uncapacitated Facility Location Problem (UFLP) with
stochastic customer demands. Each open facility has a capacity C_i.
The chance constraint requires:

    P(sum_{j assigned to i} D_j <= C_i) >= 1 - alpha,  for all i

where D_j is the random demand of customer j.

Complexity: NP-hard (generalizes deterministic UFLP).

References:
    - Bertsimas, D. & Sim, M. (2004). The price of robustness. Oper. Res.,
      52(1), 35-53. https://doi.org/10.1287/opre.1030.0065
    - Snyder, L.V. (2006). Facility location under uncertainty. IIE Trans.,
      38(7), 547-564. https://doi.org/10.1080/07408170500216480
"""
from __future__ import annotations

import sys
import os
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


@dataclass
class CCFLInstance:
    """Chance-constrained facility location instance.

    Args:
        n_facilities: Number of candidate facility sites.
        n_customers: Number of customers.
        fixed_costs: Opening cost for each facility (m,).
        assignment_costs: Cost to serve customer j from facility i (m, n).
        capacities: Capacity of each facility (m,).
        demand_scenarios: Demand for each customer under each scenario (S, n).
        alpha: Maximum allowed violation probability per facility.
        probabilities: Scenario probabilities. Uniform if None.
    """
    n_facilities: int
    n_customers: int
    fixed_costs: np.ndarray
    assignment_costs: np.ndarray
    capacities: np.ndarray
    demand_scenarios: np.ndarray
    alpha: float = 0.1
    probabilities: np.ndarray | None = None

    def __post_init__(self):
        self.fixed_costs = np.asarray(self.fixed_costs, dtype=float)
        self.assignment_costs = np.asarray(self.assignment_costs, dtype=float)
        self.capacities = np.asarray(self.capacities, dtype=float)
        self.demand_scenarios = np.asarray(self.demand_scenarios, dtype=float)
        if self.probabilities is None:
            S = self.demand_scenarios.shape[0]
            self.probabilities = np.full(S, 1.0 / S)
        else:
            self.probabilities = np.asarray(self.probabilities, dtype=float)

    @property
    def n_scenarios(self) -> int:
        return self.demand_scenarios.shape[0]

    @property
    def mean_demands(self) -> np.ndarray:
        return np.dot(self.probabilities, self.demand_scenarios)

    def total_cost(self, open_facilities: list[int],
                   assignments: np.ndarray) -> float:
        """Compute total cost (fixed + assignment).

        Args:
            open_facilities: List of open facility indices.
            assignments: Assignment matrix (m, n) or vector (n,) of facility indices.

        Returns:
            Total cost.
        """
        fixed = sum(self.fixed_costs[i] for i in open_facilities)
        if assignments.ndim == 1:
            # Vector of facility indices
            assign_cost = sum(
                self.assignment_costs[assignments[j], j]
                for j in range(self.n_customers)
            )
        else:
            assign_cost = float(np.sum(self.assignment_costs * assignments))
        return fixed + assign_cost

    def capacity_violation_prob(self, facility: int,
                                assigned_customers: list[int]) -> float:
        """Probability that demand exceeds capacity for a facility.

        Args:
            facility: Facility index.
            assigned_customers: List of customer indices assigned to it.

        Returns:
            P(sum D_j > C_i).
        """
        if not assigned_customers:
            return 0.0
        cap = self.capacities[facility]
        loads = self.demand_scenarios[:, assigned_customers].sum(axis=1)
        violated = loads > cap + 1e-9
        return float(np.dot(violated, self.probabilities))

    def is_feasible(self, open_facilities: list[int],
                    assignments: np.ndarray) -> tuple[bool, list[str]]:
        """Check if solution satisfies chance constraints.

        Args:
            open_facilities: Open facility indices.
            assignments: Customer-to-facility assignment vector (n,).

        Returns:
            (feasible, list of violation messages).
        """
        msgs = []
        for i in open_facilities:
            customers = [j for j in range(self.n_customers) if assignments[j] == i]
            viol_prob = self.capacity_violation_prob(i, customers)
            if viol_prob > self.alpha + 1e-9:
                msgs.append(f"Facility {i}: P(violation)={viol_prob:.3f} > alpha={self.alpha}")

        # Check all customers assigned to open facilities
        for j in range(self.n_customers):
            if assignments[j] not in open_facilities:
                msgs.append(f"Customer {j} assigned to closed facility {assignments[j]}")

        return len(msgs) == 0, msgs

    @classmethod
    def random(cls, n_facilities: int = 5, n_customers: int = 8,
               n_scenarios: int = 20, seed: int = 42) -> CCFLInstance:
        """Generate a random chance-constrained facility location instance."""
        rng = np.random.default_rng(seed)
        fixed_costs = rng.uniform(50, 200, n_facilities)
        assignment_costs = rng.uniform(2, 20, (n_facilities, n_customers))
        mean_demands = rng.uniform(5, 30, n_customers)
        capacities = np.full(n_facilities, mean_demands.sum() / (n_facilities * 0.4))
        demand_scenarios = np.array([
            np.maximum(0, mean_demands + rng.normal(0, 5, n_customers))
            for _ in range(n_scenarios)
        ])
        return cls(
            n_facilities=n_facilities, n_customers=n_customers,
            fixed_costs=fixed_costs, assignment_costs=assignment_costs,
            capacities=capacities, demand_scenarios=demand_scenarios,
            alpha=0.1,
        )


@dataclass
class CCFLSolution:
    """Solution to the chance-constrained facility location problem.

    Args:
        open_facilities: List of open facility indices.
        assignments: Customer-to-facility assignment vector.
        total_cost: Fixed + assignment cost.
        max_violation_prob: Maximum violation probability across facilities.
    """
    open_facilities: list[int]
    assignments: np.ndarray
    total_cost: float
    max_violation_prob: float

    def __repr__(self) -> str:
        return (f"CCFLSolution(open={self.open_facilities}, "
                f"cost={self.total_cost:.1f}, "
                f"max_viol={self.max_violation_prob:.3f})")
