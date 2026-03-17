"""
Robust Shortest Path Problem

Given a directed graph with uncertain edge weights, find the path from
source to target that minimizes the worst-case cost or worst-case regret
across all scenarios.

Two main criteria:
    - Min-Max Cost: min_{path P} max_{s in S} cost_s(P)
    - Min-Max Regret: min_{path P} max_{s in S} [cost_s(P) - cost_s(P*_s)]
      where P*_s is the optimal path under scenario s.

Complexity:
    - Min-max cost with discrete scenarios: solvable in O(S * (V+E) log V)
    - Min-max regret: NP-hard in general (Averbakh & Lebedev, 2004)

References:
    - Kouvelis, P. & Yu, G. (1997). Robust Discrete Optimization and Its
      Applications. Springer. https://doi.org/10.1007/978-1-4757-2620-6
    - Averbakh, I. & Lebedev, V. (2004). Interval data minmax regret network
      optimization problems. DAM, 138(3), 289-301.
      https://doi.org/10.1016/S0166-218X(03)00462-1
    - Bertsimas, D. & Sim, M. (2003). Robust discrete optimization and network
      flows. Math. Program., 98(1-3), 49-71.
      https://doi.org/10.1007/s10107-003-0396-4
"""
from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


@dataclass
class RobustSPInstance:
    """Robust shortest path instance with scenario-dependent edge weights.

    Args:
        n_nodes: Number of nodes (0-indexed).
        edges: List of (u, v) tuples.
        weight_scenarios: Array of shape (S, E) — weight of each edge
            under each scenario.
        source: Source node.
        target: Target node.
        probabilities: Optional scenario probabilities (for expected cost).
    """
    n_nodes: int
    edges: list[tuple[int, int]]
    weight_scenarios: np.ndarray
    source: int
    target: int
    probabilities: np.ndarray | None = None

    def __post_init__(self):
        self.weight_scenarios = np.asarray(self.weight_scenarios, dtype=float)
        if self.probabilities is None:
            S = self.weight_scenarios.shape[0]
            self.probabilities = np.full(S, 1.0 / S)
        else:
            self.probabilities = np.asarray(self.probabilities, dtype=float)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def n_scenarios(self) -> int:
        return self.weight_scenarios.shape[0]

    def adjacency_list(self, scenario: int) -> dict[int, list[tuple[int, float]]]:
        """Build adjacency list for a given scenario.

        Args:
            scenario: Scenario index.

        Returns:
            Dict mapping node to list of (neighbor, weight).
        """
        adj: dict[int, list[tuple[int, float]]] = {i: [] for i in range(self.n_nodes)}
        for e_idx, (u, v) in enumerate(self.edges):
            w = self.weight_scenarios[scenario, e_idx]
            adj[u].append((v, w))
        return adj

    def path_cost(self, path: list[int], scenario: int) -> float:
        """Compute cost of a path under a given scenario.

        Args:
            path: Sequence of node indices.
            scenario: Scenario index.

        Returns:
            Total path cost.
        """
        edge_map = {}
        for e_idx, (u, v) in enumerate(self.edges):
            edge_map[(u, v)] = e_idx

        cost = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if (u, v) not in edge_map:
                return float("inf")
            cost += self.weight_scenarios[scenario, edge_map[(u, v)]]
        return cost

    def max_cost(self, path: list[int]) -> float:
        """Worst-case cost of path across all scenarios."""
        return max(self.path_cost(path, s) for s in range(self.n_scenarios))

    def expected_cost(self, path: list[int]) -> float:
        """Expected cost of path across scenarios."""
        return sum(
            self.probabilities[s] * self.path_cost(path, s)
            for s in range(self.n_scenarios)
        )

    @classmethod
    def random(cls, n_nodes: int = 6, density: float = 0.4,
               n_scenarios: int = 5, seed: int = 42) -> RobustSPInstance:
        """Generate a random robust shortest path instance.

        Args:
            n_nodes: Number of nodes.
            density: Edge density.
            n_scenarios: Number of weight scenarios.
            seed: Random seed.

        Returns:
            Random RobustSPInstance.
        """
        rng = np.random.default_rng(seed)
        edges = []
        for u in range(n_nodes):
            for v in range(n_nodes):
                if u != v and rng.random() < density:
                    edges.append((u, v))

        # Ensure at least one path from 0 to n_nodes-1
        if not edges:
            for i in range(n_nodes - 1):
                edges.append((i, i + 1))

        E = len(edges)
        # Generate base weights + perturbation per scenario
        base = rng.uniform(1, 10, E)
        weight_scenarios = np.array([
            np.maximum(1, base + rng.uniform(-3, 5, E))
            for _ in range(n_scenarios)
        ])

        return cls(
            n_nodes=n_nodes,
            edges=edges,
            weight_scenarios=weight_scenarios,
            source=0,
            target=n_nodes - 1,
        )


@dataclass
class RobustSPSolution:
    """Solution to a robust shortest path problem.

    Args:
        path: Sequence of nodes from source to target.
        max_cost: Worst-case cost across scenarios.
        expected_cost: Expected cost across scenarios.
        scenario_costs: Cost under each scenario.
        max_regret: Worst-case regret (if computed).
    """
    path: list[int]
    max_cost: float
    expected_cost: float
    scenario_costs: list[float] = field(default_factory=list)
    max_regret: float | None = None

    def __repr__(self) -> str:
        regret_str = f", regret={self.max_regret:.2f}" if self.max_regret is not None else ""
        return (f"RobustSPSolution(path={self.path}, "
                f"max_cost={self.max_cost:.2f}, "
                f"E[cost]={self.expected_cost:.2f}{regret_str})")


if __name__ == "__main__":
    inst = RobustSPInstance.random(n_nodes=5, n_scenarios=4)
    print(f"Instance: {inst.n_nodes} nodes, {inst.n_edges} edges, "
          f"{inst.n_scenarios} scenarios")
    print(f"Edges: {inst.edges}")
    print(f"Weight scenarios shape: {inst.weight_scenarios.shape}")
