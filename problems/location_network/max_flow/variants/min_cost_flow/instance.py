"""
Minimum Cost Flow Problem — Instance and Solution.

Given a directed graph with edge capacities and per-unit costs, a source
with supply b(s) > 0 and a sink with demand b(t) < 0, find the flow that
satisfies supply/demand while minimizing total cost.

Complexity: Polynomial — O(V^2 * E) via successive shortest paths.

References:
    Ahuja, R.K., Magnanti, T.L. & Orlin, J.B. (1993). Network Flows:
    Theory, Algorithms, and Applications. Prentice Hall.
    ISBN 978-0136175490.

    Goldberg, A.V. & Tarjan, R.E. (1990). Finding minimum-cost
    circulations by successive approximation. Mathematics of Operations
    Research, 15(3), 430-466. https://doi.org/10.1287/moor.15.3.430
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class MinCostFlowInstance:
    """Minimum Cost Flow instance.

    Attributes:
        n: Number of nodes.
        edges: List of (from, to, capacity, cost) tuples.
        supply: Supply/demand per node, shape (n,). Positive = supply,
                negative = demand. Must sum to zero.
        name: Optional instance name.
    """

    n: int
    edges: list[tuple[int, int, float, float]]
    supply: np.ndarray
    name: str = ""

    def __post_init__(self):
        self.supply = np.asarray(self.supply, dtype=float)

    @classmethod
    def random(
        cls,
        n: int = 6,
        edge_prob: float = 0.4,
        cap_range: tuple[int, int] = (5, 20),
        cost_range: tuple[int, int] = (1, 10),
        total_supply: float = 15.0,
        seed: int | None = None,
    ) -> MinCostFlowInstance:
        rng = np.random.default_rng(seed)
        edges = []
        # Ensure path from 0 to n-1
        for i in range(n - 1):
            cap = float(rng.integers(cap_range[0], cap_range[1] + 1))
            cost = float(rng.integers(cost_range[0], cost_range[1] + 1))
            edges.append((i, i + 1, cap, cost))
        # Extra edges
        for i in range(n):
            for j in range(n):
                if i != j and j != i + 1 and rng.random() < edge_prob:
                    cap = float(rng.integers(cap_range[0], cap_range[1] + 1))
                    cost = float(rng.integers(cost_range[0], cost_range[1] + 1))
                    edges.append((i, j, cap, cost))
        supply = np.zeros(n)
        supply[0] = total_supply
        supply[-1] = -total_supply
        return cls(n=n, edges=edges, supply=supply, name=f"random_{n}")


@dataclass
class MinCostFlowSolution:
    """Minimum Cost Flow solution.

    Attributes:
        flow: Dict mapping (from, to) to flow value.
        total_cost: Total transportation cost.
        total_flow: Total flow from source.
    """

    flow: dict[tuple[int, int], float]
    total_cost: float
    total_flow: float

    def __repr__(self) -> str:
        return f"MinCostFlowSolution(cost={self.total_cost:.1f}, flow={self.total_flow:.1f})"


def validate_solution(
    instance: MinCostFlowInstance, solution: MinCostFlowSolution
) -> tuple[bool, list[str]]:
    errors = []

    # Check capacity constraints
    edge_caps = {}
    for u, v, cap, cost in instance.edges:
        edge_caps[(u, v)] = cap

    for (u, v), f in solution.flow.items():
        if f < -1e-6:
            errors.append(f"Negative flow on ({u},{v}): {f:.4f}")
        cap = edge_caps.get((u, v), 0.0)
        if f > cap + 1e-6:
            errors.append(f"Flow ({u},{v}): {f:.4f} > capacity {cap:.4f}")

    # Check flow conservation
    for node in range(instance.n):
        outflow = sum(solution.flow.get((node, v), 0.0) for v in range(instance.n))
        inflow = sum(solution.flow.get((u, node), 0.0) for u in range(instance.n))
        balance = outflow - inflow
        expected = instance.supply[node]
        if abs(balance - expected) > 1e-4:
            errors.append(
                f"Node {node}: balance {balance:.4f} != supply {expected:.4f}")

    # Check cost
    actual_cost = 0.0
    edge_costs = {(u, v): c for u, v, _, c in instance.edges}
    for (u, v), f in solution.flow.items():
        if f > 1e-8:
            actual_cost += f * edge_costs.get((u, v), 0.0)
    if abs(actual_cost - solution.total_cost) > 1e-2:
        errors.append(f"Cost: {solution.total_cost:.2f} != {actual_cost:.2f}")

    return len(errors) == 0, errors


def small_mcf_4() -> MinCostFlowInstance:
    return MinCostFlowInstance(
        n=4,
        edges=[
            (0, 1, 10, 2), (0, 2, 8, 5),
            (1, 2, 5, 1), (1, 3, 7, 3),
            (2, 3, 10, 4),
        ],
        supply=np.array([10, 0, 0, -10], dtype=float),
        name="small_4",
    )


if __name__ == "__main__":
    inst = small_mcf_4()
    print(f"{inst.name}: n={inst.n}, supply={inst.supply}")
