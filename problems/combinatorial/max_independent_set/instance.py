"""
Maximum Independent Set Problem (MIS)

Given an undirected graph G = (V, E), find the largest subset S of
vertices such that no two vertices in S are adjacent.

Equivalent to Maximum Clique on the complement graph.
alpha(G) + chi(G) >= n (Ramsey theory relationship).

Complexity: NP-hard. Inapproximable within n^{1-epsilon}.

References:
    - Garey, M.R. & Johnson, D.S. (1979). Computers and Intractability.
      W.H. Freeman, p. 194.
    - Boppana, R. & Halldórsson, M.M. (1992). Approximating maximum
      independent sets by excluding subgraphs. BIT, 32(2), 180-196.
      https://doi.org/10.1007/BF01994876
"""
from __future__ import annotations

import sys
import os
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


@dataclass
class MISInstance:
    """Maximum independent set problem instance.

    Args:
        n_vertices: Number of vertices.
        edges: List of (u, v) undirected edges.
    """
    n_vertices: int
    edges: list[tuple[int, int]]

    def adjacency_list(self) -> dict[int, set[int]]:
        adj: dict[int, set[int]] = {v: set() for v in range(self.n_vertices)}
        for u, v in self.edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def is_independent(self, subset: list[int]) -> bool:
        s = set(subset)
        for u, v in self.edges:
            if u in s and v in s:
                return False
        return True

    @classmethod
    def random(cls, n_vertices: int = 20, density: float = 0.3,
               seed: int = 42) -> MISInstance:
        rng = np.random.default_rng(seed)
        edges = []
        for u in range(n_vertices):
            for v in range(u + 1, n_vertices):
                if rng.random() < density:
                    edges.append((u, v))
        return cls(n_vertices=n_vertices, edges=edges)

    @classmethod
    def cycle(cls, n: int) -> MISInstance:
        """Cycle graph: MIS size = floor(n/2)."""
        edges = [(i, (i + 1) % n) for i in range(n)]
        return cls(n_vertices=n, edges=edges)


@dataclass
class MISSolution:
    independent_set: list[int]
    size: int
    is_valid: bool

    def __repr__(self) -> str:
        return f"MISSolution(size={self.size}, valid={self.is_valid})"
