"""
Chinese Postman Problem (CPP) — Instance and Solution.

Problem notation: CPP (undirected | all edges | min total weight)

Given an undirected weighted graph, find the minimum-weight closed walk
(tour) that traverses every edge at least once. If the graph is Eulerian
(all vertices have even degree), the optimal tour has weight equal to the
sum of all edge weights. Otherwise, edges must be duplicated to make all
vertices even-degree.

Complexity: Polynomial for undirected graphs — O(V^3) via minimum weight
perfect matching on odd-degree vertices.

References:
    Edmonds, J. & Johnson, E.L. (1973). Matching, Euler tours and the
    Chinese postman. Mathematical Programming, 5(1), 88-124.
    https://doi.org/10.1007/BF01580113

    Kwan, M.K. (1962). Graphic programming using odd or even points.
    Chinese Mathematics, 1(1), 273-277.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class ChinesePostmanInstance:
    """Chinese Postman Problem instance (undirected weighted graph).

    Attributes:
        n_vertices: Number of vertices.
        edges: List of (u, v, weight) tuples. Vertices are 0-indexed.
        adj_matrix: Adjacency/weight matrix (n_vertices x n_vertices).
            adj_matrix[i][j] = weight of edge (i,j), 0 if no edge.
        name: Optional instance name.
    """

    n_vertices: int
    edges: list[tuple[int, int, float]]
    adj_matrix: np.ndarray
    name: str = ""

    def __post_init__(self):
        self.adj_matrix = np.asarray(self.adj_matrix, dtype=float)
        if self.adj_matrix.shape != (self.n_vertices, self.n_vertices):
            raise ValueError(
                f"adj_matrix shape {self.adj_matrix.shape} "
                f"!= ({self.n_vertices}, {self.n_vertices})"
            )

    def degree(self, v: int) -> int:
        """Return the degree of vertex v."""
        d = 0
        for u, w, _ in self.edges:
            if u == v or w == v:
                d += 1
        return d

    def odd_degree_vertices(self) -> list[int]:
        """Return list of vertices with odd degree."""
        degrees = [0] * self.n_vertices
        for u, v, _ in self.edges:
            degrees[u] += 1
            degrees[v] += 1
        return [v for v in range(self.n_vertices) if degrees[v] % 2 == 1]

    def is_eulerian(self) -> bool:
        """Check if the graph is Eulerian (all vertices have even degree)."""
        return len(self.odd_degree_vertices()) == 0

    def is_connected(self) -> bool:
        """Check if the graph is connected using BFS."""
        if self.n_vertices == 0:
            return True
        visited = set()
        queue = [0]
        visited.add(0)
        while queue:
            v = queue.pop(0)
            for u, w, _ in self.edges:
                neighbor = None
                if u == v:
                    neighbor = w
                elif w == v:
                    neighbor = u
                if neighbor is not None and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return len(visited) == self.n_vertices

    def total_edge_weight(self) -> float:
        """Return the sum of all edge weights."""
        return sum(w for _, _, w in self.edges)

    def shortest_paths(self) -> np.ndarray:
        """Compute all-pairs shortest paths using Floyd-Warshall.

        Returns:
            (n_vertices x n_vertices) distance matrix.
        """
        n = self.n_vertices
        dist = np.full((n, n), float("inf"))
        np.fill_diagonal(dist, 0.0)

        for u, v, w in self.edges:
            dist[u][v] = min(dist[u][v], w)
            dist[v][u] = min(dist[v][u], w)

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]

        return dist

    @classmethod
    def random(
        cls,
        n_vertices: int,
        edge_prob: float = 0.5,
        weight_range: tuple[float, float] = (1.0, 20.0),
        seed: int | None = None,
    ) -> ChinesePostmanInstance:
        """Generate a random connected undirected weighted graph.

        Args:
            n_vertices: Number of vertices.
            edge_prob: Probability of edge between any two vertices.
            weight_range: Range for random edge weights.
            seed: Random seed for reproducibility.

        Returns:
            A random ChinesePostmanInstance (guaranteed connected).
        """
        rng = np.random.default_rng(seed)
        adj_matrix = np.zeros((n_vertices, n_vertices))
        edges = []

        # First, create a spanning tree to ensure connectivity
        vertices = list(range(n_vertices))
        rng.shuffle(vertices)
        for i in range(1, n_vertices):
            u = vertices[i]
            v = vertices[rng.integers(0, i)]
            w = round(rng.uniform(weight_range[0], weight_range[1]), 1)
            edges.append((min(u, v), max(u, v), w))
            adj_matrix[u][v] = w
            adj_matrix[v][u] = w

        # Add random extra edges
        for i in range(n_vertices):
            for j in range(i + 1, n_vertices):
                if adj_matrix[i][j] == 0 and rng.random() < edge_prob:
                    w = round(rng.uniform(weight_range[0], weight_range[1]), 1)
                    edges.append((i, j, w))
                    adj_matrix[i][j] = w
                    adj_matrix[j][i] = w

        return cls(
            n_vertices=n_vertices,
            edges=edges,
            adj_matrix=adj_matrix,
            name=f"random_cpp_{n_vertices}",
        )

    @classmethod
    def from_edges(
        cls,
        n_vertices: int,
        edges: list[tuple[int, int, float]],
        name: str = "",
    ) -> ChinesePostmanInstance:
        """Create instance from edge list.

        Args:
            n_vertices: Number of vertices.
            edges: List of (u, v, weight) tuples.
            name: Optional instance name.

        Returns:
            A ChinesePostmanInstance.
        """
        adj_matrix = np.zeros((n_vertices, n_vertices))
        for u, v, w in edges:
            adj_matrix[u][v] = w
            adj_matrix[v][u] = w
        return cls(
            n_vertices=n_vertices,
            edges=edges,
            adj_matrix=adj_matrix,
            name=name,
        )


@dataclass
class ChinesePostmanSolution:
    """Solution to a Chinese Postman Problem.

    Attributes:
        tour: List of vertex indices forming the closed walk.
            Starts and ends at the same vertex.
        total_weight: Total weight of the tour.
        duplicated_edges: List of edges that were duplicated
            (empty if graph was Eulerian).
    """

    tour: list[int]
    total_weight: float
    duplicated_edges: list[tuple[int, int, float]] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"ChinesePostmanSolution(total_weight={self.total_weight:.2f}, "
            f"tour_length={len(self.tour)}, "
            f"duplicated={len(self.duplicated_edges)})"
        )


def validate_solution(
    instance: ChinesePostmanInstance, solution: ChinesePostmanSolution
) -> tuple[bool, list[str]]:
    """Validate a Chinese Postman solution.

    Args:
        instance: The CPP instance.
        solution: The solution to validate.

    Returns:
        Tuple of (is_valid, list of error messages).
    """
    errors = []

    tour = solution.tour

    # Check tour is a closed walk
    if len(tour) < 2:
        errors.append("Tour must have at least 2 vertices")
        return False, errors
    if tour[0] != tour[-1]:
        errors.append("Tour must start and end at the same vertex")

    # Check all vertices are valid
    for v in tour:
        if v < 0 or v >= instance.n_vertices:
            errors.append(f"Invalid vertex {v}")

    # Check all edges in tour are valid (exist in graph)
    tour_edges: list[tuple[int, int]] = []
    for i in range(len(tour) - 1):
        u, v = tour[i], tour[i + 1]
        tour_edges.append((min(u, v), max(u, v)))

    # Build edge multiset from instance + duplicated edges
    available: dict[tuple[int, int], int] = {}
    for u, v, _ in instance.edges:
        key = (min(u, v), max(u, v))
        available[key] = available.get(key, 0) + 1
    for u, v, _ in solution.duplicated_edges:
        key = (min(u, v), max(u, v))
        available[key] = available.get(key, 0) + 1

    # Count tour edge usage
    usage: dict[tuple[int, int], int] = {}
    for e in tour_edges:
        usage[e] = usage.get(e, 0) + 1

    # Check all original edges are covered
    for u, v, _ in instance.edges:
        key = (min(u, v), max(u, v))
        if key not in usage or usage[key] < 1:
            errors.append(f"Edge ({u}, {v}) not traversed")

    return len(errors) == 0, errors


# ── Benchmark instances ──────────────────────────────────────────────────────


def eulerian_square() -> ChinesePostmanInstance:
    """4-vertex Eulerian graph (square). All vertices degree 2.

    0 -- 1
    |    |
    3 -- 2
    """
    edges = [
        (0, 1, 10.0),
        (1, 2, 10.0),
        (2, 3, 10.0),
        (0, 3, 10.0),
    ]
    return ChinesePostmanInstance.from_edges(4, edges, "eulerian_square")


def non_eulerian_triangle() -> ChinesePostmanInstance:
    """3-vertex complete graph (triangle). All vertices degree 2 => Eulerian.

    Actually K3 has all vertices degree 2, so it IS Eulerian.
    """
    edges = [
        (0, 1, 5.0),
        (1, 2, 7.0),
        (0, 2, 3.0),
    ]
    return ChinesePostmanInstance.from_edges(3, edges, "triangle_k3")


def bridge_graph() -> ChinesePostmanInstance:
    """5-vertex graph with a bridge. Vertices 1 and 2 have odd degree.

    0 -- 1 -- 2 -- 3
              |
              4
    """
    edges = [
        (0, 1, 4.0),
        (1, 2, 3.0),
        (2, 3, 5.0),
        (2, 4, 6.0),
    ]
    return ChinesePostmanInstance.from_edges(5, edges, "bridge_graph")


if __name__ == "__main__":
    inst = eulerian_square()
    print(f"eulerian_square: {inst.n_vertices} vertices, "
          f"{len(inst.edges)} edges, Eulerian={inst.is_eulerian()}")

    inst2 = bridge_graph()
    print(f"bridge_graph: {inst2.n_vertices} vertices, "
          f"{len(inst2.edges)} edges, Eulerian={inst2.is_eulerian()}")
    print(f"  Odd-degree vertices: {inst2.odd_degree_vertices()}")
