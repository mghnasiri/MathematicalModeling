"""Multi-Objective Shortest Path Problem.

Problem: Given a directed graph where each edge has a vector of costs
(e.g., time and monetary cost), find all Pareto-optimal paths from
source to target.

Complexity: NP-hard in general (number of Pareto-optimal paths can be
exponential). Polynomial for fixed number of objectives with bounded
edge costs.

References:
    Hansen, P. (1980). Bicriterion path problems. In Multiple Criteria
    Decision Making Theory and Application (pp. 109-127). Springer.

    Martins, E. Q. V. (1984). On a multicriteria shortest path problem.
    European Journal of Operational Research, 16(2), 236-245.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class MultiObjectiveSPInstance:
    """Multi-objective shortest path problem instance.

    Args:
        n: Number of nodes (0-indexed).
        n_objectives: Number of cost objectives per edge.
        edges: List of (from, to, costs) where costs is a tuple of floats.
        source: Source node index.
        target: Target node index.
    """
    n: int
    n_objectives: int
    edges: list[tuple[int, int, tuple[float, ...]]]
    source: int
    target: int

    @classmethod
    def random(cls, n: int = 8, n_objectives: int = 2,
               density: float = 0.4, seed: int = 42,
               source: int = 0, target: int | None = None
               ) -> MultiObjectiveSPInstance:
        """Generate a random multi-objective shortest path instance.

        Args:
            n: Number of nodes.
            n_objectives: Number of objectives.
            density: Edge density.
            seed: Random seed.
            source: Source node.
            target: Target node (defaults to n-1).

        Returns:
            A random MultiObjectiveSPInstance.
        """
        if target is None:
            target = n - 1
        rng = np.random.default_rng(seed)
        edges = []
        # Ensure connectivity: create a path from source to target
        perm = list(range(n))
        rng.shuffle(perm)
        # Make sure source is first and target is last in the path
        perm.remove(source)
        perm.remove(target)
        path = [source] + list(perm) + [target]
        for i in range(len(path) - 1):
            costs = tuple(float(rng.integers(1, 11)) for _ in range(n_objectives))
            edges.append((path[i], path[i + 1], costs))

        # Add random edges
        for i in range(n):
            for j in range(n):
                if i != j and rng.random() < density:
                    # Check if edge already exists
                    if not any(e[0] == i and e[1] == j for e in edges):
                        costs = tuple(float(rng.integers(1, 11)) for _ in range(n_objectives))
                        edges.append((i, j, costs))

        return cls(n=n, n_objectives=n_objectives, edges=edges,
                   source=source, target=target)

    def get_adjacency(self) -> dict[int, list[tuple[int, tuple[float, ...]]]]:
        """Build adjacency list representation.

        Returns:
            Dict mapping node to list of (neighbor, costs).
        """
        adj: dict[int, list[tuple[int, tuple[float, ...]]]] = {i: [] for i in range(self.n)}
        for u, v, costs in self.edges:
            adj[u].append((v, costs))
        return adj


@dataclass
class MOSPSolution:
    """Solution to a multi-objective shortest path problem.

    Args:
        pareto_paths: List of Pareto-optimal paths. Each path is a list
            of node indices from source to target.
        pareto_costs: List of cost vectors for each Pareto-optimal path.
    """
    pareto_paths: list[list[int]]
    pareto_costs: list[tuple[float, ...]]

    def __repr__(self) -> str:
        return (f"MOSPSolution(n_pareto_paths={len(self.pareto_paths)}, "
                f"costs={self.pareto_costs})")
