"""
Two-Stage Stochastic Programming (2SSP)

In the first stage, decisions x are made before uncertainty is revealed.
In the second stage, recourse decisions y(s) are made for each scenario s
to compensate for constraint violations.

    min  c^T x + E_s[q(s)^T y(s)]
    s.t. Ax = b                          (first-stage constraints)
         T(s)x + W(s)y(s) = h(s)         (second-stage, per scenario s)
         x >= 0, y(s) >= 0

With discrete scenarios, the expectation becomes a weighted sum:
    min  c^T x + sum_s p_s * q(s)^T y(s)

This is the standard formulation from Birge & Louveaux (2011).

Complexity: NP-hard in general. The deterministic equivalent LP has
            size O(n1 + S * n2) variables and O(m1 + S * m2) constraints.

References:
    - Birge, J.R. & Louveaux, F. (2011). Introduction to Stochastic
      Programming, 2nd ed. Springer. https://doi.org/10.1007/978-1-4614-0237-4
    - Dantzig, G.B. (1955). Linear programming under uncertainty.
      Management Science, 1(3-4), 197-206. https://doi.org/10.1287/mnsc.1.3-4.197
"""
from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


@dataclass
class TwoStageSPInstance:
    """Two-stage stochastic linear program with discrete scenarios.

    Args:
        c: First-stage cost vector (n1,).
        A: First-stage constraint matrix (m1, n1).
        b: First-stage RHS (m1,).
        scenarios: List of scenario dicts, each with keys:
            'q': second-stage cost (n2,)
            'T': technology matrix (m2, n1)
            'W': recourse matrix (m2, n2)
            'h': second-stage RHS (m2,)
        probabilities: Scenario probabilities (S,). Uniform if None.
    """
    c: np.ndarray
    A: np.ndarray
    b: np.ndarray
    scenarios: list[dict]
    probabilities: np.ndarray | None = None

    def __post_init__(self):
        self.c = np.asarray(self.c, dtype=float)
        self.A = np.asarray(self.A, dtype=float)
        self.b = np.asarray(self.b, dtype=float)
        if self.probabilities is None:
            S = len(self.scenarios)
            self.probabilities = np.full(S, 1.0 / S)
        else:
            self.probabilities = np.asarray(self.probabilities, dtype=float)

    @property
    def n1(self) -> int:
        """Number of first-stage variables."""
        return len(self.c)

    @property
    def n2(self) -> int:
        """Number of second-stage variables (from first scenario)."""
        return len(self.scenarios[0]["q"])

    @property
    def n_scenarios(self) -> int:
        return len(self.scenarios)

    @property
    def m1(self) -> int:
        """Number of first-stage constraints."""
        return len(self.b)

    @property
    def m2(self) -> int:
        """Number of second-stage constraints (per scenario)."""
        return len(self.scenarios[0]["h"])

    @classmethod
    def newsvendor_as_2ssp(cls, unit_cost: float, selling_price: float,
                           salvage_value: float,
                           demand_scenarios: np.ndarray) -> TwoStageSPInstance:
        """Formulate newsvendor as a two-stage stochastic program.

        First stage: x = order quantity
        Second stage: y1 = units sold, y2 = units unsold

        min c*x + E[-p*y1 - v*y2]
        s.t. y1 + y2 = x           (inventory balance)
             y1 <= D(s)             (can't sell more than demand)
             x, y1, y2 >= 0

        Args:
            unit_cost: Purchase cost per unit.
            selling_price: Revenue per unit sold.
            salvage_value: Revenue per unsold unit.
            demand_scenarios: Array of demand values.

        Returns:
            TwoStageSPInstance.
        """
        c = np.array([unit_cost])
        A = np.zeros((0, 1))
        b = np.zeros(0)

        scenarios = []
        for d in demand_scenarios:
            scenarios.append({
                "q": np.array([-selling_price, -salvage_value]),
                "T": np.array([[-1.0, 0.0]]),  # -x + y1 + y2 = 0
                "W": np.array([[1.0, 1.0]]),
                "h": np.array([0.0]),
            })

        return cls(c=c, A=A, b=b, scenarios=scenarios)

    @classmethod
    def capacity_planning(cls, n_facilities: int = 3, n_scenarios: int = 5,
                          seed: int = 42) -> TwoStageSPInstance:
        """Generate a capacity planning 2SSP instance.

        First stage: build capacity x_i at facility i (cost c_i per unit).
        Second stage: allocate production y_{ij}(s) from facility i to
        customer j under demand scenario s.

        Args:
            n_facilities: Number of facilities.
            n_scenarios: Number of demand scenarios.
            seed: Random seed.

        Returns:
            TwoStageSPInstance.
        """
        rng = np.random.default_rng(seed)
        n_customers = n_facilities  # square for simplicity

        # First stage: capacity costs
        c = rng.uniform(5, 15, n_facilities)
        A = np.zeros((0, n_facilities))
        b_fs = np.zeros(0)

        n2 = n_facilities * n_customers  # production allocation variables

        scenarios = []
        for _ in range(n_scenarios):
            demands = rng.uniform(10, 50, n_customers)
            q = rng.uniform(1, 5, n2)  # production costs

            # Constraints: capacity (sum_j y_ij <= x_i) and demand (sum_i y_ij >= d_j)
            m2 = n_facilities + n_customers
            T = np.zeros((m2, n_facilities))
            W = np.zeros((m2, n2))
            h = np.zeros(m2)

            # Capacity constraints: -x_i + sum_j y_ij <= 0
            for i in range(n_facilities):
                T[i, i] = -1.0
                for j in range(n_customers):
                    W[i, i * n_customers + j] = 1.0

            # Demand constraints: sum_i y_ij >= d_j  =>  -sum_i y_ij <= -d_j
            for j in range(n_customers):
                for i in range(n_facilities):
                    W[n_facilities + j, i * n_customers + j] = -1.0
                h[n_facilities + j] = -demands[j]

            scenarios.append({"q": q, "T": T, "W": W, "h": h})

        return cls(c=c, A=A, b=b_fs, scenarios=scenarios)


@dataclass
class TwoStageSPSolution:
    """Solution to a two-stage stochastic program.

    Args:
        x: First-stage decision vector.
        first_stage_cost: c^T x.
        expected_recourse_cost: E[q(s)^T y(s)].
        total_cost: first_stage_cost + expected_recourse_cost.
        recourse_solutions: Optional dict mapping scenario index to y(s).
    """
    x: np.ndarray
    first_stage_cost: float
    expected_recourse_cost: float
    total_cost: float
    recourse_solutions: dict[int, np.ndarray] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"TwoStageSPSolution(x={self.x}, "
                f"1st={self.first_stage_cost:.2f}, "
                f"E[2nd]={self.expected_recourse_cost:.2f}, "
                f"total={self.total_cost:.2f})")


if __name__ == "__main__":
    inst = TwoStageSPInstance.capacity_planning(n_facilities=3, n_scenarios=5)
    print(f"Capacity planning: {inst.n1} first-stage vars, "
          f"{inst.n2} second-stage vars, {inst.n_scenarios} scenarios")
    print(f"First-stage costs: {inst.c}")
