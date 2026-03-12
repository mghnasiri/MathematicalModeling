"""
Maximum Clique Problem — Instance and Solution definitions.

Problem notation: MC (max clique)

Given an undirected graph G = (V, E), find the largest complete subgraph
(clique) — a subset S of V where every pair of vertices in S is connected.

Complexity: NP-hard (Karp, 1972). No polynomial-time constant-factor
approximation unless P=NP.

References:
    Bron, C. & Kerbosch, J. (1973). Finding all cliques of an undirected
    graph. Communications of the ACM, 16(9), 575-577.
    https://doi.org/10.1145/362342.362367

    Tomita, E., Tanaka, A. & Takahashi, H. (2006). The worst-case time
    complexity for generating all maximal cliques. Theoretical Computer
    Science, 363(1), 28-42.
    https://doi.org/10.1016/j.tcs.2006.06.015
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class MaxCliqueInstance:
    """Maximum Clique Problem instance.

    Attributes:
        n_vertices: Number of vertices.
        edges: List of (u, v) tuples (0-indexed).
        adj: Adjacency set per vertex.
        name: Optional instance name.
    """

    n_vertices: int
    edges: list[tuple[int, int]]
    name: str = ""

    def __post_init__(self):
        self.adj: dict[int, set[int]] = {v: set() for v in range(self.n_vertices)}
        for u, v in self.edges:
            self.adj[u].add(v)
            self.adj[v].add(u)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    def is_clique(self, vertices: list[int] | set[int]) -> bool:
        """Check if a set of vertices forms a clique."""
        vlist = list(vertices)
        for i in range(len(vlist)):
            for j in range(i + 1, len(vlist)):
                if vlist[j] not in self.adj[vlist[i]]:
                    return False
        return True

    @classmethod
    def random(
        cls,
        n_vertices: int = 10,
        density: float = 0.5,
        seed: int | None = None,
    ) -> MaxCliqueInstance:
        rng = np.random.default_rng(seed)
        edges = []
        for i in range(n_vertices):
            for j in range(i + 1, n_vertices):
                if rng.random() < density:
                    edges.append((i, j))
        return cls(n_vertices=n_vertices, edges=edges, name=f"random_mc_{n_vertices}")

    @classmethod
    def complete(cls, n: int) -> MaxCliqueInstance:
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        return cls(n_vertices=n, edges=edges, name=f"K_{n}")

    @classmethod
    def petersen(cls) -> MaxCliqueInstance:
        """Petersen graph: 10 vertices, max clique = 2."""
        outer = [(i, (i + 1) % 5) for i in range(5)]
        inner = [(5 + i, 5 + (i + 2) % 5) for i in range(5)]
        spokes = [(i, i + 5) for i in range(5)]
        return cls(n_vertices=10, edges=outer + inner + spokes, name="petersen")


@dataclass
class MaxCliqueSolution:
    """Solution to a Maximum Clique Problem.

    Attributes:
        clique: List of vertex indices in the clique.
        size: Size of the clique.
    """

    clique: list[int]
    size: int

    def __repr__(self) -> str:
        return f"MaxCliqueSolution(size={self.size}, clique={self.clique})"


if __name__ == "__main__":
    inst = MaxCliqueInstance.petersen()
    print(f"Petersen: {inst.n_vertices} vertices, {inst.n_edges} edges")
