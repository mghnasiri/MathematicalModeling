"""Hub Location Problem — p-Hub Median.

Given n nodes, a flow matrix W[i][j] representing demand between origin i
and destination j, a distance matrix D[i][j], and a number p of hubs to open,
select p hub nodes and assign each non-hub node to a hub to minimize total
transportation cost.

In the single-allocation p-hub median, each node is assigned to exactly one hub.
Flows are routed: origin -> origin's hub -> destination's hub -> destination.
Inter-hub links enjoy a discount factor alpha (0 < alpha < 1).

Complexity: NP-hard (O'Kelly, 1987).

References:
    O'Kelly, M. E. (1987). A quadratic integer program for the location of
    interacting hub facilities. European Journal of Operational Research, 32(3),
    393-404.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HubLocationInstance:
    """Single-allocation p-hub median problem instance.

    Attributes:
        n: Number of nodes.
        p: Number of hubs to open.
        flows: Flow matrix W[i][j], shape (n, n).
        distances: Distance matrix D[i][j], shape (n, n).
        alpha: Inter-hub discount factor (0 < alpha < 1).
    """

    n: int
    p: int
    flows: np.ndarray
    distances: np.ndarray
    alpha: float

    @classmethod
    def random(cls, n: int = 10, p: int = 3, alpha: float = 0.75,
               seed: int | None = None) -> HubLocationInstance:
        """Generate a random hub location instance.

        Args:
            n: Number of nodes.
            p: Number of hubs.
            alpha: Inter-hub discount factor.
            seed: Random seed for reproducibility.

        Returns:
            A random HubLocationInstance.
        """
        rng = np.random.default_rng(seed)
        # Random node positions in [0, 100]^2
        positions = rng.uniform(0, 100, size=(n, 2))
        # Euclidean distance matrix
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        distances = np.sqrt((diff ** 2).sum(axis=2))
        # Random flow matrix (non-negative, zero diagonal)
        flows = rng.uniform(0, 100, size=(n, n))
        np.fill_diagonal(flows, 0)
        return cls(n=n, p=p, flows=flows, distances=distances, alpha=alpha)

    def transport_cost(self, hubs: list[int], assignments: list[int]) -> float:
        """Compute total transportation cost.

        Args:
            hubs: List of hub node indices.
            assignments: assignments[i] = hub index assigned to node i.

        Returns:
            Total transportation cost.
        """
        total = 0.0
        for i in range(self.n):
            for j in range(self.n):
                if self.flows[i, j] == 0:
                    continue
                h_i = assignments[i]
                h_j = assignments[j]
                cost = (self.distances[i, h_i]
                        + self.alpha * self.distances[h_i, h_j]
                        + self.distances[h_j, j])
                total += self.flows[i, j] * cost
        return total


@dataclass
class HubLocationSolution:
    """Solution to a hub location problem.

    Attributes:
        hubs: List of hub node indices.
        assignments: assignments[i] = hub index assigned to node i.
        objective: Total transportation cost.
    """

    hubs: list[int]
    assignments: list[int]
    objective: float

    def __repr__(self) -> str:
        return (f"HubLocationSolution(hubs={self.hubs}, "
                f"objective={self.objective:.2f})")
