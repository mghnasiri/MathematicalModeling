"""Graph Partitioning Problem — balanced k-way partitioning.

Problem: Partition the vertices of a weighted graph into k roughly equal
subsets to minimize the total weight of edges between different partitions
(edge cut).

Complexity: NP-hard even for k=2 (graph bisection).

References:
    Kernighan, B. W., & Lin, S. (1970). An efficient heuristic procedure
    for partitioning graphs. The Bell System Technical Journal, 49(2), 291-307.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class GraphPartitioningInstance:
    """Graph partitioning problem instance.

    Args:
        n: Number of vertices.
        k: Number of partitions.
        adjacency: Symmetric weight matrix (n x n). adjacency[i][j] is
            the edge weight between vertices i and j (0 if no edge).
        balance_tolerance: Maximum allowed deviation from perfect balance
            as a fraction. Partition size must be in
            [floor(n/k), ceil(n/k) + tolerance * n].
    """
    n: int
    k: int
    adjacency: np.ndarray
    balance_tolerance: float = 0.1

    @classmethod
    def random(cls, n: int = 20, k: int = 2, density: float = 0.3,
               seed: int = 42, balance_tolerance: float = 0.1
               ) -> GraphPartitioningInstance:
        """Generate a random graph partitioning instance.

        Args:
            n: Number of vertices.
            k: Number of partitions.
            density: Edge density (probability of edge existence).
            seed: Random seed.
            balance_tolerance: Balance tolerance fraction.

        Returns:
            A random GraphPartitioningInstance.
        """
        rng = np.random.default_rng(seed)
        adj = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < density:
                    w = rng.integers(1, 11)
                    adj[i, j] = w
                    adj[j, i] = w
        return cls(n=n, k=k, adjacency=adj, balance_tolerance=balance_tolerance)

    def edge_cut(self, partition: list[int]) -> float:
        """Compute total weight of edges crossing partition boundaries.

        Args:
            partition: List of length n, partition[i] is the partition
                label (0 to k-1) for vertex i.

        Returns:
            Total edge cut weight.
        """
        cut = 0.0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if partition[i] != partition[j]:
                    cut += self.adjacency[i, j]
        return cut

    def is_balanced(self, partition: list[int]) -> bool:
        """Check if partition satisfies balance constraint.

        Args:
            partition: Partition assignment for each vertex.

        Returns:
            True if all partition sizes are within tolerance.
        """
        counts = np.bincount(partition, minlength=self.k)
        ideal = self.n / self.k
        max_size = int(np.ceil(ideal) + self.balance_tolerance * self.n)
        min_size = max(1, int(np.floor(ideal) - self.balance_tolerance * self.n))
        return all(min_size <= c <= max_size for c in counts)


@dataclass
class GraphPartitioningSolution:
    """Solution to a graph partitioning problem.

    Args:
        partition: List of partition labels (0 to k-1) for each vertex.
        edge_cut: Total weight of edges between different partitions.
    """
    partition: list[int]
    edge_cut: float

    def __repr__(self) -> str:
        sizes = {}
        for p in self.partition:
            sizes[p] = sizes.get(p, 0) + 1
        return (f"GraphPartitioningSolution(edge_cut={self.edge_cut:.1f}, "
                f"sizes={dict(sorted(sizes.items()))})")
