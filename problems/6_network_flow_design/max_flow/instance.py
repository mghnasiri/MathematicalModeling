"""
Maximum Flow Problem — Instance and Solution definitions.

Problem notation: Max-Flow

Given a directed graph G = (V, E) with edge capacities c(u,v), a source
node s, and a sink node t, find the maximum flow from s to t such that
flow on each edge does not exceed its capacity, and flow conservation
holds at every node except s and t.

The Max-Flow Min-Cut Theorem (Ford & Fulkerson, 1956) states that the
maximum flow equals the minimum cut capacity.

Complexity:
- Edmonds-Karp: O(V * E^2)
- Push-relabel: O(V^2 * E)
- Dinic's: O(V^2 * E)

References:
    Ford, L.R. & Fulkerson, D.R. (1956). Maximal flow through a
    network. Canadian Journal of Mathematics, 8, 399-404.
    https://doi.org/10.4153/CJM-1956-045-5

    Edmonds, J. & Karp, R.M. (1972). Theoretical improvements in
    algorithmic efficiency for network flow problems. Journal of the
    ACM, 19(2), 248-264.
    https://doi.org/10.1145/321694.321699

    Goldberg, A.V. & Tarjan, R.E. (1988). A new approach to the
    maximum-flow problem. Journal of the ACM, 35(4), 921-940.
    https://doi.org/10.1145/48014.61051
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class MaxFlowInstance:
    """Maximum Flow problem instance.

    Attributes:
        n: Number of nodes (0-indexed).
        source: Source node index.
        sink: Sink node index.
        edges: List of (u, v, capacity) tuples.
        capacity_matrix: Adjacency capacity matrix, shape (n, n).
        name: Optional instance name.
    """

    n: int
    source: int
    sink: int
    edges: list[tuple[int, int, float]]
    capacity_matrix: np.ndarray = field(repr=False)
    name: str = ""

    def __post_init__(self):
        self.capacity_matrix = np.asarray(self.capacity_matrix, dtype=float)
        if self.capacity_matrix.shape != (self.n, self.n):
            raise ValueError(
                f"capacity_matrix shape {self.capacity_matrix.shape} != ({self.n}, {self.n})"
            )

    @classmethod
    def from_edges(
        cls, n: int, source: int, sink: int,
        edges: list[tuple[int, int, float]], name: str = ""
    ) -> MaxFlowInstance:
        """Create instance from edge list.

        Args:
            n: Number of nodes.
            source: Source node.
            sink: Sink node.
            edges: List of (u, v, capacity).
            name: Instance name.

        Returns:
            MaxFlowInstance.
        """
        cap = np.zeros((n, n))
        for u, v, c in edges:
            cap[u][v] += c
        return cls(
            n=n, source=source, sink=sink,
            edges=edges, capacity_matrix=cap, name=name,
        )

    @classmethod
    def random(
        cls, n: int, density: float = 0.3,
        cap_range: tuple[float, float] = (1.0, 20.0),
        seed: int | None = None,
    ) -> MaxFlowInstance:
        """Generate a random max-flow instance.

        Source = 0, sink = n-1.

        Args:
            n: Number of nodes.
            density: Edge probability.
            cap_range: Capacity range.
            seed: Random seed.

        Returns:
            A random MaxFlowInstance.
        """
        rng = np.random.default_rng(seed)
        edges = []
        cap = np.zeros((n, n))

        for u in range(n):
            for v in range(n):
                if u != v and rng.random() < density:
                    c = round(rng.uniform(cap_range[0], cap_range[1]), 1)
                    edges.append((u, v, c))
                    cap[u][v] = c

        return cls(
            n=n, source=0, sink=n - 1,
            edges=edges, capacity_matrix=cap, name=f"random_{n}",
        )


@dataclass
class MaxFlowSolution:
    """Solution to a maximum flow problem.

    Attributes:
        max_flow: Maximum flow value from source to sink.
        flow_matrix: Flow on each edge, shape (n, n).
        min_cut: Optional tuple (S, T) — the minimum cut partition.
    """

    max_flow: float
    flow_matrix: np.ndarray
    min_cut: tuple[list[int], list[int]] | None = None

    def __repr__(self) -> str:
        return f"MaxFlowSolution(max_flow={self.max_flow:.1f})"


def validate_solution(
    instance: MaxFlowInstance, solution: MaxFlowSolution
) -> tuple[bool, list[str]]:
    """Validate a max-flow solution."""
    errors = []
    n = instance.n
    flow = solution.flow_matrix

    # Check capacity constraints
    for u in range(n):
        for v in range(n):
            if flow[u][v] > instance.capacity_matrix[u][v] + 1e-6:
                errors.append(
                    f"Flow {flow[u][v]:.2f} > capacity "
                    f"{instance.capacity_matrix[u][v]:.2f} on edge ({u},{v})"
                )
            if flow[u][v] < -1e-6:
                errors.append(f"Negative flow {flow[u][v]:.2f} on edge ({u},{v})")

    # Check flow conservation (except source and sink)
    for v in range(n):
        if v == instance.source or v == instance.sink:
            continue
        inflow = sum(flow[u][v] for u in range(n))
        outflow = sum(flow[v][u] for u in range(n))
        if abs(inflow - outflow) > 1e-6:
            errors.append(
                f"Flow conservation violated at node {v}: "
                f"in={inflow:.2f}, out={outflow:.2f}"
            )

    # Verify max_flow value
    source_out = sum(flow[instance.source][v] for v in range(n))
    if abs(source_out - solution.max_flow) > 1e-4:
        errors.append(
            f"Reported flow {solution.max_flow:.2f} != "
            f"source outflow {source_out:.2f}"
        )

    return len(errors) == 0, errors


# ── Benchmark instances ──────────────────────────────────────────────────────


def simple_flow_4() -> MaxFlowInstance:
    """4-node network. Max flow = 26.

    0 -> 1 (16), 0 -> 2 (13)
    1 -> 2 (4),  1 -> 3 (12)
    2 -> 1 (10), 2 -> 3 (14)
    """
    edges = [
        (0, 1, 16), (0, 2, 13),
        (1, 2, 4),  (1, 3, 12),
        (2, 1, 10), (2, 3, 14),
    ]
    return MaxFlowInstance.from_edges(4, 0, 3, edges, name="simple4")


def two_path_flow() -> MaxFlowInstance:
    """5-node network with two paths. Max flow = 5.

    Path 1: 0->1->4 (cap 3)
    Path 2: 0->2->3->4 (cap 2)
    """
    edges = [
        (0, 1, 3), (1, 4, 3),
        (0, 2, 2), (2, 3, 2), (3, 4, 2),
    ]
    return MaxFlowInstance.from_edges(5, 0, 4, edges, name="two_path5")


if __name__ == "__main__":
    inst = simple_flow_4()
    print(f"{inst.name}: n={inst.n}, source={inst.source}, sink={inst.sink}")
    print(f"  edges: {inst.edges}")
