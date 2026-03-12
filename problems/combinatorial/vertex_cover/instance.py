"""
Minimum Vertex Cover Problem — Instance and Solution definitions.

Problem notation: MVC (min vertex cover)

Given an undirected graph G = (V, E), find the smallest subset S of V
such that every edge has at least one endpoint in S.

Complexity: NP-hard (Karp, 1972). Greedy 2-approximation exists.
Relation: complement of Maximum Independent Set.

References:
    Karp, R.M. (1972). Reducibility among combinatorial problems.
    In Complexity of Computer Computations, Plenum Press.

    Bar-Yehuda, R. & Even, S. (1981). A linear-time approximation
    algorithm for the weighted vertex cover problem. Journal of
    Algorithms, 2(2), 198-203.
    https://doi.org/10.1016/0196-6774(81)90021-7
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class VertexCoverInstance:
    """Minimum Vertex Cover Problem instance.

    Attributes:
        n_vertices: Number of vertices.
        edges: List of (u, v) tuples (0-indexed).
        name: Optional instance name.
    """

    n_vertices: int
    edges: list[tuple[int, int]]
    name: str = ""

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    def is_vertex_cover(self, cover: list[int] | set[int]) -> bool:
        """Check if a set of vertices is a valid vertex cover."""
        cover_set = set(cover)
        return all(u in cover_set or v in cover_set for u, v in self.edges)

    @classmethod
    def random(
        cls,
        n_vertices: int = 10,
        density: float = 0.3,
        seed: int | None = None,
    ) -> VertexCoverInstance:
        rng = np.random.default_rng(seed)
        edges = []
        for i in range(n_vertices):
            for j in range(i + 1, n_vertices):
                if rng.random() < density:
                    edges.append((i, j))
        return cls(n_vertices=n_vertices, edges=edges, name=f"random_vc_{n_vertices}")

    @classmethod
    def cycle(cls, n: int) -> VertexCoverInstance:
        """Cycle graph C_n. MVC size = ceil(n/2)."""
        edges = [(i, (i + 1) % n) for i in range(n)]
        return cls(n_vertices=n, edges=edges, name=f"cycle_{n}")

    @classmethod
    def complete(cls, n: int) -> VertexCoverInstance:
        """Complete graph K_n. MVC size = n-1."""
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        return cls(n_vertices=n, edges=edges, name=f"K_{n}")

    @classmethod
    def star(cls, n: int) -> VertexCoverInstance:
        """Star graph S_n. Center=0, leaves=1..n. MVC size = 1."""
        edges = [(0, i) for i in range(1, n + 1)]
        return cls(n_vertices=n + 1, edges=edges, name=f"star_{n}")


@dataclass
class VertexCoverSolution:
    """Solution to a Vertex Cover Problem.

    Attributes:
        cover: Set of vertex indices in the cover.
        size: Size of the cover.
    """

    cover: list[int]
    size: int

    def __repr__(self) -> str:
        return f"VertexCoverSolution(size={self.size}, cover={self.cover})"


if __name__ == "__main__":
    inst = VertexCoverInstance.cycle(6)
    print(f"Cycle(6): {inst.n_vertices} vertices, {inst.n_edges} edges")
    print(f"  {[0,2,4]} is cover: {inst.is_vertex_cover([0,2,4])}")
