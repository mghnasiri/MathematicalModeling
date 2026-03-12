"""
All-Pairs Shortest Path (APSP) — Instance and Solution.

Compute shortest paths between all pairs of vertices in a weighted
directed graph. Handles negative weights (but not negative cycles).

Complexity: Floyd-Warshall O(V^3), Johnson's O(V^2 log V + VE).

References:
    Floyd, R.W. (1962). Algorithm 97: Shortest path. Communications
    of the ACM, 5(6), 345. https://doi.org/10.1145/367766.368168

    Warshall, S. (1962). A theorem on boolean matrices. Journal of the
    ACM, 9(1), 11-12. https://doi.org/10.1145/321105.321107

    Johnson, D.B. (1977). Efficient algorithms for shortest paths in
    sparse networks. Journal of the ACM, 24(1), 1-13.
    https://doi.org/10.1145/321992.321993
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class APSPInstance:
    """All-Pairs Shortest Path instance.

    Attributes:
        n: Number of nodes.
        weight_matrix: Weight matrix, shape (n, n). INF for no edge.
        name: Optional instance name.
    """

    n: int
    weight_matrix: np.ndarray
    name: str = ""

    def __post_init__(self):
        self.weight_matrix = np.asarray(self.weight_matrix, dtype=float)

    @classmethod
    def from_edges(cls, n: int, edges: list[tuple[int, int, float]],
                   name: str = "") -> APSPInstance:
        """Create from edge list."""
        W = np.full((n, n), np.inf)
        np.fill_diagonal(W, 0.0)
        for u, v, w in edges:
            W[u][v] = w
        return cls(n=n, weight_matrix=W, name=name)

    @classmethod
    def random(
        cls,
        n: int = 5,
        density: float = 0.5,
        weight_range: tuple[int, int] = (1, 20),
        seed: int | None = None,
    ) -> APSPInstance:
        rng = np.random.default_rng(seed)
        W = np.full((n, n), np.inf)
        np.fill_diagonal(W, 0.0)
        for i in range(n):
            for j in range(n):
                if i != j and rng.random() < density:
                    W[i][j] = float(rng.integers(weight_range[0],
                                                  weight_range[1] + 1))
        return cls(n=n, weight_matrix=W, name=f"random_apsp_{n}")


@dataclass
class APSPSolution:
    """APSP solution.

    Attributes:
        dist_matrix: Shortest distance matrix, shape (n, n).
        next_matrix: Next-hop matrix for path reconstruction.
            next_matrix[i][j] = next node on shortest path from i to j.
            -1 if no path exists.
    """

    dist_matrix: np.ndarray
    next_matrix: np.ndarray

    def __repr__(self) -> str:
        n = self.dist_matrix.shape[0]
        reachable = np.sum(self.dist_matrix < np.inf) - n
        return f"APSPSolution(n={n}, reachable_pairs={reachable})"

    def get_path(self, source: int, target: int) -> list[int] | None:
        """Reconstruct shortest path from source to target."""
        if self.dist_matrix[source][target] == np.inf:
            return None
        path = [source]
        current = source
        while current != target:
            current = int(self.next_matrix[current][target])
            if current == -1:
                return None
            path.append(current)
        return path


def validate_solution(
    instance: APSPInstance, solution: APSPSolution
) -> tuple[bool, list[str]]:
    errors = []
    n = instance.n

    if solution.dist_matrix.shape != (n, n):
        errors.append(f"Wrong shape: {solution.dist_matrix.shape}")
        return False, errors

    # Check diagonal
    for i in range(n):
        if abs(solution.dist_matrix[i][i]) > 1e-6:
            errors.append(f"dist[{i}][{i}] = {solution.dist_matrix[i][i]:.2f} != 0")

    # Check triangle inequality
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if (solution.dist_matrix[i][k] + solution.dist_matrix[k][j] <
                        solution.dist_matrix[i][j] - 1e-6):
                    errors.append(
                        f"Triangle inequality violated: "
                        f"d({i},{k}) + d({k},{j}) < d({i},{j})")
                    if len(errors) > 5:
                        return False, errors

    # Check path reconstruction
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if solution.dist_matrix[i][j] < np.inf:
                path = solution.get_path(i, j)
                if path is None:
                    errors.append(f"No path {i}->{j} but dist is finite")
                elif path[0] != i or path[-1] != j:
                    errors.append(f"Path {i}->{j} endpoints wrong")

    return len(errors) == 0, errors


def small_apsp_4() -> APSPInstance:
    """Small 4-node graph for APSP."""
    return APSPInstance.from_edges(
        n=4,
        edges=[
            (0, 1, 3), (0, 2, 8), (1, 2, 2),
            (1, 3, 5), (2, 3, 1), (3, 0, 4),
        ],
        name="small_apsp_4",
    )


if __name__ == "__main__":
    inst = small_apsp_4()
    print(f"{inst.name}: n={inst.n}")
    print(f"Weight matrix:\n{inst.weight_matrix}")
