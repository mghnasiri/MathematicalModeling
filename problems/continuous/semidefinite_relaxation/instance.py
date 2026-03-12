"""SDP Relaxation for MAX-CUT Problem.

Problem: Given a weighted undirected graph, partition vertices into two
sets S and V\\S to maximize the total weight of edges crossing the cut.

The Goemans-Williamson SDP relaxation achieves an expected approximation
ratio of alpha_GW >= 0.878.

Complexity: MAX-CUT is NP-hard (Karp, 1972).

References:
    Goemans, M. X., & Williamson, D. P. (1995). Improved approximation
    algorithms for maximum cut and satisfiability problems using
    semidefinite programming. Journal of the ACM, 42(6), 1115-1145.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class MaxCutInstance:
    """MAX-CUT problem instance.

    Args:
        n: Number of vertices.
        adjacency: Symmetric weight matrix (n x n). adjacency[i][j] is
            the edge weight between vertices i and j (0 if no edge).
    """
    n: int
    adjacency: np.ndarray

    @classmethod
    def random(cls, n: int = 10, density: float = 0.5,
               seed: int = 42) -> MaxCutInstance:
        """Generate a random MAX-CUT instance.

        Args:
            n: Number of vertices.
            density: Edge density.
            seed: Random seed.

        Returns:
            A random MaxCutInstance.
        """
        rng = np.random.default_rng(seed)
        adj = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < density:
                    w = float(rng.integers(1, 11))
                    adj[i, j] = w
                    adj[j, i] = w
        return cls(n=n, adjacency=adj)

    def cut_value(self, partition: list[int]) -> float:
        """Compute the cut value for a given partition.

        Args:
            partition: List of length n with values in {0, 1} or {-1, 1}.
                Vertices are split into two sets.

        Returns:
            Total weight of edges crossing the cut.
        """
        # Normalize to {0, 1}
        labels = [0 if p <= 0 else 1 for p in partition]
        cut = 0.0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if labels[i] != labels[j]:
                    cut += self.adjacency[i, j]
        return cut

    def total_edge_weight(self) -> float:
        """Return total weight of all edges."""
        return float(np.sum(self.adjacency) / 2)


@dataclass
class MaxCutSolution:
    """Solution to a MAX-CUT problem.

    Args:
        partition: List of labels (0 or 1) for each vertex.
        cut_value: Total weight of edges in the cut.
        sdp_bound: Upper bound from SDP relaxation (if computed).
    """
    partition: list[int]
    cut_value: float
    sdp_bound: float | None = None

    def __repr__(self) -> str:
        s0 = sum(1 for p in self.partition if p == 0)
        s1 = sum(1 for p in self.partition if p == 1)
        bound_str = f", sdp_bound={self.sdp_bound:.1f}" if self.sdp_bound else ""
        return (f"MaxCutSolution(cut={self.cut_value:.1f}, "
                f"sizes=({s0}, {s1}){bound_str})")
