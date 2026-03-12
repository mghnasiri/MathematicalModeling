"""
Graph Coloring Problem (GCP)

Given an undirected graph G = (V, E), assign a color to each vertex such
that no two adjacent vertices share the same color, minimizing the number
of colors used (chromatic number chi(G)).

Complexity: NP-hard to determine chi(G). NP-hard to approximate within
n^{1-epsilon} for any epsilon > 0.

References:
    - Brélaz, D. (1979). New methods to color the vertices of a graph.
      Comm. ACM, 22(4), 251-256. https://doi.org/10.1145/359094.359101
    - Jensen, T.R. & Toft, B. (2011). Graph Coloring Problems. Wiley.
"""
from __future__ import annotations

import sys
import os
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


@dataclass
class GraphColoringInstance:
    """Graph coloring problem instance.

    Args:
        n_vertices: Number of vertices.
        edges: List of (u, v) undirected edges.
    """
    n_vertices: int
    edges: list[tuple[int, int]]

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    def adjacency_list(self) -> dict[int, set[int]]:
        adj: dict[int, set[int]] = {v: set() for v in range(self.n_vertices)}
        for u, v in self.edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def is_valid_coloring(self, colors: list[int]) -> bool:
        for u, v in self.edges:
            if colors[u] == colors[v]:
                return False
        return True

    def n_colors_used(self, colors: list[int]) -> int:
        return len(set(colors))

    @classmethod
    def random(cls, n_vertices: int = 20, density: float = 0.3,
               seed: int = 42) -> GraphColoringInstance:
        rng = np.random.default_rng(seed)
        edges = []
        for u in range(n_vertices):
            for v in range(u + 1, n_vertices):
                if rng.random() < density:
                    edges.append((u, v))
        return cls(n_vertices=n_vertices, edges=edges)

    @classmethod
    def petersen(cls) -> GraphColoringInstance:
        """Petersen graph: chi = 3."""
        edges = [(0,1),(0,4),(0,5),(1,2),(1,6),(2,3),(2,7),
                 (3,4),(3,8),(4,9),(5,7),(5,8),(6,8),(6,9),(7,9)]
        return cls(n_vertices=10, edges=edges)

    @classmethod
    def cycle(cls, n: int) -> GraphColoringInstance:
        """Cycle graph: chi=2 if even, chi=3 if odd."""
        edges = [(i, (i + 1) % n) for i in range(n)]
        return cls(n_vertices=n, edges=edges)


@dataclass
class GraphColoringSolution:
    colors: list[int]
    n_colors: int
    is_valid: bool

    def __repr__(self) -> str:
        return (f"GraphColoringSolution(n_colors={self.n_colors}, "
                f"valid={self.is_valid})")
