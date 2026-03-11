"""
Steiner Tree Problem in Graphs — Instance and Solution.

Given an undirected weighted graph G = (V, E) and a subset S ⊆ V of
terminal nodes, find the minimum-weight tree that spans all terminals.
Non-terminal (Steiner) nodes may be included if they reduce total weight.

Complexity: NP-hard (Karp, 1972). Polynomial for |S| = 2 (shortest path)
and for |S| = |V| (MST).

References:
    Hwang, F.K., Richards, D.S. & Winter, P. (1992). The Steiner Tree
    Problem. Annals of Discrete Mathematics, 53. North-Holland.

    Kou, L., Markowsky, G. & Berman, L. (1981). A fast algorithm for
    Steiner trees. Acta Informatica, 15(2), 141-145.
    https://doi.org/10.1007/BF00288961
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class SteinerTreeInstance:
    """Steiner Tree instance.

    Attributes:
        n: Number of vertices.
        edges: List of (u, v, weight) tuples.
        terminals: Set of terminal vertex indices.
        name: Optional instance name.
    """

    n: int
    edges: list[tuple[int, int, float]]
    terminals: set[int]
    name: str = ""

    def adjacency_matrix(self) -> np.ndarray:
        """Weighted adjacency matrix (inf for no edge)."""
        adj = np.full((self.n, self.n), np.inf)
        np.fill_diagonal(adj, 0.0)
        for u, v, w in self.edges:
            adj[u][v] = min(adj[u][v], w)
            adj[v][u] = min(adj[v][u], w)
        return adj

    def shortest_paths(self) -> np.ndarray:
        """All-pairs shortest paths via Floyd-Warshall."""
        dist = self.adjacency_matrix()
        for k in range(self.n):
            for i in range(self.n):
                for j in range(self.n):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        return dist

    @classmethod
    def random(
        cls,
        n: int = 10,
        n_terminals: int = 4,
        edge_prob: float = 0.4,
        weight_range: tuple[int, int] = (1, 20),
        seed: int | None = None,
    ) -> SteinerTreeInstance:
        rng = np.random.default_rng(seed)
        edges = []
        # Ensure connectivity via a random spanning tree first
        perm = rng.permutation(n).tolist()
        for i in range(1, n):
            w = int(rng.integers(weight_range[0], weight_range[1] + 1))
            edges.append((perm[i - 1], perm[i], float(w)))
        # Add random extra edges
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < edge_prob:
                    w = int(rng.integers(weight_range[0], weight_range[1] + 1))
                    edges.append((i, j, float(w)))
        terminals = set(rng.choice(n, size=min(n_terminals, n), replace=False).tolist())
        return cls(n=n, edges=edges, terminals=terminals, name=f"random_{n}")


@dataclass
class SteinerTreeSolution:
    """Steiner Tree solution.

    Attributes:
        tree_edges: List of (u, v, weight) in the Steiner tree.
        total_weight: Total edge weight.
    """

    tree_edges: list[tuple[int, int, float]]
    total_weight: float

    def __repr__(self) -> str:
        return f"SteinerTreeSolution(edges={len(self.tree_edges)}, weight={self.total_weight:.1f})"


def validate_solution(
    instance: SteinerTreeInstance, solution: SteinerTreeSolution
) -> tuple[bool, list[str]]:
    errors = []

    # Check connectivity of terminals
    if not solution.tree_edges and len(instance.terminals) > 1:
        errors.append("No edges but multiple terminals")
        return False, errors

    # Build adjacency from tree edges
    adj: dict[int, set[int]] = {}
    weight_sum = 0.0
    for u, v, w in solution.tree_edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
        weight_sum += w

    if abs(weight_sum - solution.total_weight) > 1e-4:
        errors.append(f"Weight mismatch: {solution.total_weight:.2f} != {weight_sum:.2f}")

    # BFS from first terminal to check all terminals reachable
    if instance.terminals:
        start = next(iter(instance.terminals))
        visited = set()
        queue = [start]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            for nb in adj.get(node, set()):
                if nb not in visited:
                    queue.append(nb)
        for t in instance.terminals:
            if t not in visited:
                errors.append(f"Terminal {t} not reachable in tree")

    return len(errors) == 0, errors


def small_steiner_6() -> SteinerTreeInstance:
    return SteinerTreeInstance(
        n=6,
        edges=[
            (0, 1, 2), (0, 2, 5), (1, 2, 3), (1, 3, 4),
            (2, 3, 1), (2, 4, 6), (3, 4, 3), (3, 5, 7),
            (4, 5, 2),
        ],
        terminals={0, 4, 5},
        name="small_6",
    )


if __name__ == "__main__":
    inst = small_steiner_6()
    print(f"{inst.name}: n={inst.n}, terminals={inst.terminals}")
