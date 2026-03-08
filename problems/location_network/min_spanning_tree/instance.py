"""
Minimum Spanning Tree Problem — Instance and Solution definitions.

Problem notation: MST

Given an undirected, weighted, connected graph G = (V, E), find the
spanning tree T of minimum total edge weight — a connected acyclic
subgraph spanning all vertices.

Complexity:
- Kruskal's: O(E log E) with union-find
- Prim's: O(E log V) with binary heap

References:
    Kruskal, J.B. (1956). On the shortest spanning subtree of a graph
    and the traveling salesman problem. Proceedings of the American
    Mathematical Society, 7(1), 48-50.
    https://doi.org/10.1090/S0002-9939-1956-0078686-7

    Prim, R.C. (1957). Shortest connection networks and some
    generalizations. Bell System Technical Journal, 36(6), 1389-1401.
    https://doi.org/10.1002/j.1538-7305.1957.tb01515.x

    Cormen, T.H., Leiserson, C.E., Rivest, R.L. & Stein, C. (2009).
    Introduction to Algorithms. 3rd ed. MIT Press, Chapter 23.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class MSTInstance:
    """Minimum Spanning Tree problem instance (undirected graph).

    Attributes:
        n: Number of nodes.
        edges: List of (u, v, weight) tuples (undirected).
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
                self.adjacency[v].append((u, w))

    @classmethod
    def from_edges(
        cls, n: int, edges: list[tuple[int, int, float]], name: str = ""
    ) -> MSTInstance:
        """Create instance from undirected edge list.

        Args:
            n: Number of nodes.
            edges: List of (u, v, weight).
            name: Instance name.

        Returns:
            MSTInstance.
        """
        adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
        for u, v, w in edges:
            adj[u].append((v, w))
            adj[v].append((u, w))
        return cls(n=n, edges=edges, adjacency=adj, name=name)

    @classmethod
    def from_matrix(
        cls, matrix: np.ndarray, name: str = ""
    ) -> MSTInstance:
        """Create instance from symmetric adjacency matrix.

        Args:
            matrix: (n, n) weight matrix. inf = no edge.
            name: Instance name.

        Returns:
            MSTInstance.
        """
        n = matrix.shape[0]
        edges = []
        adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if matrix[u][v] < float("inf"):
                    w = float(matrix[u][v])
                    edges.append((u, v, w))
                    adj[u].append((v, w))
                    adj[v].append((u, w))
        return cls(n=n, edges=edges, adjacency=adj, name=name)

    @classmethod
    def random(
        cls, n: int, density: float = 0.5,
        weight_range: tuple[float, float] = (1.0, 20.0),
        seed: int | None = None,
    ) -> MSTInstance:
        """Generate a random connected undirected graph.

        Ensures connectivity by first creating a random spanning tree,
        then adding edges with given density.

        Args:
            n: Number of nodes.
            density: Additional edge probability.
            weight_range: Edge weight range.
            seed: Random seed.

        Returns:
            A random MSTInstance.
        """
        rng = np.random.default_rng(seed)
        edges = []
        edge_set: set[tuple[int, int]] = set()

        # Random spanning tree for connectivity
        perm = rng.permutation(n)
        for i in range(1, n):
            u, v = int(perm[i - 1]), int(perm[i])
            if u > v:
                u, v = v, u
            w = round(rng.uniform(weight_range[0], weight_range[1]), 1)
            edges.append((u, v, w))
            edge_set.add((u, v))

        # Additional random edges
        for u in range(n):
            for v in range(u + 1, n):
                if (u, v) not in edge_set and rng.random() < density:
                    w = round(rng.uniform(weight_range[0], weight_range[1]), 1)
                    edges.append((u, v, w))
                    edge_set.add((u, v))

        return cls.from_edges(n, edges, name=f"random_{n}")


@dataclass
class MSTSolution:
    """Solution to a MST problem.

    Attributes:
        tree_edges: List of (u, v, weight) in the spanning tree.
        total_weight: Sum of edge weights.
    """

    tree_edges: list[tuple[int, int, float]]
    total_weight: float

    def __repr__(self) -> str:
        return (
            f"MSTSolution(weight={self.total_weight:.1f}, "
            f"edges={len(self.tree_edges)})"
        )


def validate_solution(
    instance: MSTInstance, solution: MSTSolution
) -> tuple[bool, list[str]]:
    """Validate a MST solution."""
    errors = []
    n = instance.n

    if len(solution.tree_edges) != n - 1:
        errors.append(
            f"Tree has {len(solution.tree_edges)} edges, expected {n - 1}"
        )

    # Check all edges exist in instance
    inst_edges = set()
    for u, v, w in instance.edges:
        inst_edges.add((min(u, v), max(u, v)))

    for u, v, w in solution.tree_edges:
        key = (min(u, v), max(u, v))
        if key not in inst_edges:
            errors.append(f"Edge ({u},{v}) not in instance")

    # Check connectivity (forms a tree)
    if not errors:
        adj: dict[int, list[int]] = {i: [] for i in range(n)}
        for u, v, _ in solution.tree_edges:
            adj[u].append(v)
            adj[v].append(u)

        visited = set()
        stack = [0]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            for neighbor in adj[node]:
                if neighbor not in visited:
                    stack.append(neighbor)

        if len(visited) != n:
            errors.append(f"Tree spans {len(visited)} nodes, expected {n}")

    # Check total weight
    actual = sum(w for _, _, w in solution.tree_edges)
    if abs(actual - solution.total_weight) > 1e-4:
        errors.append(
            f"Reported weight {solution.total_weight:.2f} != actual {actual:.2f}"
        )

    return len(errors) == 0, errors


# ── Benchmark instances ──────────────────────────────────────────────────────


def triangle_graph() -> MSTInstance:
    """3-node triangle. MST weight = 3 (edges 0-1, 1-2)."""
    edges = [(0, 1, 1), (1, 2, 2), (0, 2, 4)]
    return MSTInstance.from_edges(3, edges, name="triangle3")


def simple_graph_6() -> MSTInstance:
    """6-node graph. MST weight = 13."""
    edges = [
        (0, 1, 4), (0, 2, 3),
        (1, 2, 1), (1, 3, 2),
        (2, 3, 4), (2, 4, 5),
        (3, 4, 7), (3, 5, 2),
        (4, 5, 1),
    ]
    return MSTInstance.from_edges(6, edges, name="simple6")


if __name__ == "__main__":
    inst = simple_graph_6()
    print(f"{inst.name}: {inst.n} nodes, {len(inst.edges)} edges")
