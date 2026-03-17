"""
Shortest Path Problem — Instance and Solution definitions.

Problem notation: SPP (Shortest Path Problem)

Given a directed graph G = (V, E) with edge weights w(u,v), find the
path from source s to target t with minimum total weight.

Complexity:
- Dijkstra's algorithm: O((V + E) log V) with binary heap (non-negative weights).
- Bellman-Ford: O(V * E) (handles negative weights, detects negative cycles).

References:
    Dijkstra, E.W. (1959). A note on two problems in connexion with
    graphs. Numerische Mathematik, 1(1), 269-271.
    https://doi.org/10.1007/BF01386390

    Bellman, R. (1958). On a routing problem. Quarterly of Applied
    Mathematics, 16(1), 87-90.
    https://doi.org/10.1090/qam/102435

    Ford, L.R. & Fulkerson, D.R. (1962). Flows in Networks.
    Princeton University Press.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class ShortestPathInstance:
    """Shortest Path problem instance (directed graph).

    Attributes:
        n: Number of nodes (0-indexed).
        edges: List of (u, v, weight) tuples.
        adjacency: Adjacency list — adjacency[u] = [(v, weight), ...].
        name: Optional instance name.
    """

    n: int
    edges: list[tuple[int, int, float]]
    adjacency: list[list[tuple[int, float]]]
    name: str = ""

    def __post_init__(self):
        if not self.adjacency:
            self.adjacency = [[] for _ in range(self.n)]
            for u, v, w in self.edges:
                self.adjacency[u].append((v, w))

    @classmethod
    def from_edges(
        cls, n: int, edges: list[tuple[int, int, float]], name: str = ""
    ) -> ShortestPathInstance:
        """Create instance from edge list.

        Args:
            n: Number of nodes.
            edges: List of (u, v, weight).
            name: Instance name.

        Returns:
            ShortestPathInstance.
        """
        adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
        for u, v, w in edges:
            adj[u].append((v, w))
        return cls(n=n, edges=edges, adjacency=adj, name=name)

    @classmethod
    def from_matrix(
        cls, matrix: np.ndarray, name: str = ""
    ) -> ShortestPathInstance:
        """Create instance from adjacency matrix.

        Args:
            matrix: (n, n) weight matrix. inf = no edge.
            name: Instance name.

        Returns:
            ShortestPathInstance.
        """
        n = matrix.shape[0]
        edges = []
        adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
        for u in range(n):
            for v in range(n):
                if u != v and matrix[u][v] < float("inf"):
                    edges.append((u, v, float(matrix[u][v])))
                    adj[u].append((v, float(matrix[u][v])))
        return cls(n=n, edges=edges, adjacency=adj, name=name)

    @classmethod
    def random(
        cls,
        n: int,
        density: float = 0.3,
        weight_range: tuple[float, float] = (1.0, 20.0),
        seed: int | None = None,
    ) -> ShortestPathInstance:
        """Generate a random directed graph.

        Args:
            n: Number of nodes.
            density: Edge probability.
            weight_range: Range for edge weights.
            seed: Random seed.

        Returns:
            A random ShortestPathInstance.
        """
        rng = np.random.default_rng(seed)
        edges = []
        adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]

        for u in range(n):
            for v in range(n):
                if u != v and rng.random() < density:
                    w = round(rng.uniform(weight_range[0], weight_range[1]), 1)
                    edges.append((u, v, w))
                    adj[u].append((v, w))

        return cls(n=n, edges=edges, adjacency=adj, name=f"random_{n}")

    def has_negative_weights(self) -> bool:
        """Check if any edge has negative weight."""
        return any(w < 0 for _, _, w in self.edges)


@dataclass
class ShortestPathSolution:
    """Solution to a Shortest Path problem.

    Attributes:
        source: Source node.
        target: Target node.
        path: List of nodes from source to target.
        distance: Total path distance.
    """

    source: int
    target: int
    path: list[int]
    distance: float

    def __repr__(self) -> str:
        return (
            f"ShortestPathSolution(distance={self.distance:.1f}, "
            f"path={self.path})"
        )


def validate_solution(
    instance: ShortestPathInstance, solution: ShortestPathSolution
) -> tuple[bool, list[str]]:
    """Validate a shortest path solution."""
    errors = []

    if not solution.path:
        if solution.source != solution.target:
            errors.append("Empty path but source != target")
        return len(errors) == 0, errors

    if solution.path[0] != solution.source:
        errors.append(f"Path starts at {solution.path[0]}, expected {solution.source}")
    if solution.path[-1] != solution.target:
        errors.append(f"Path ends at {solution.path[-1]}, expected {solution.target}")

    # Check edge connectivity and compute distance
    total = 0.0
    for i in range(len(solution.path) - 1):
        u, v = solution.path[i], solution.path[i + 1]
        found = False
        for neighbor, w in instance.adjacency[u]:
            if neighbor == v:
                total += w
                found = True
                break
        if not found:
            errors.append(f"No edge from {u} to {v}")

    if not errors and abs(total - solution.distance) > 1e-6:
        errors.append(
            f"Reported distance {solution.distance:.2f} != actual {total:.2f}"
        )

    return len(errors) == 0, errors


# ── Benchmark instances ──────────────────────────────────────────────────────


def simple_graph_5() -> ShortestPathInstance:
    """5-node directed graph.

    Shortest path 0->4: 0->1->3->4, distance=7.
    """
    edges = [
        (0, 1, 2), (0, 2, 5),
        (1, 2, 1), (1, 3, 3),
        (2, 3, 4), (2, 4, 6),
        (3, 4, 2),
    ]
    return ShortestPathInstance.from_edges(5, edges, name="simple5")


def negative_weight_graph() -> ShortestPathInstance:
    """5-node graph with negative edges (no negative cycles).

    Shortest path 0->4: 0->1->2->3->4, distance=1.
    """
    edges = [
        (0, 1, 3), (0, 3, 5),
        (1, 2, -2), (1, 3, 4),
        (2, 3, 1),
        (3, 4, -1),
    ]
    return ShortestPathInstance.from_edges(5, edges, name="negative_weights")


if __name__ == "__main__":
    inst = simple_graph_5()
    print(f"{inst.name}: {inst.n} nodes, {len(inst.edges)} edges")
    for u, v, w in inst.edges:
        print(f"  {u} -> {v}: {w}")
