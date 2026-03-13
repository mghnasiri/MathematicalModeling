"""Multi-Commodity Flow Problem instance and solution definitions.

Problem: Given a directed graph with edge capacities and multiple commodities
(each with source, sink, demand), find flows for all commodities that satisfy
demands while respecting shared edge capacities.

Notation: MCFP
Complexity: Polynomial (LP formulation)

References:
    Ahuja, R. K., Magnanti, T. L., & Orlin, J. B. (1993).
    Network Flows: Theory, Algorithms, and Applications. Prentice Hall.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class Commodity:
    """A single commodity with source, sink, and demand."""
    source: int
    sink: int
    demand: float


@dataclass
class MultiCommodityFlowInstance:
    """Multi-commodity flow problem instance.

    Attributes:
        n_nodes: Number of nodes in the graph.
        edges: List of (from, to) tuples.
        capacities: Capacity of each edge (indexed same as edges).
        commodities: List of Commodity objects.
    """
    n_nodes: int
    edges: list[tuple[int, int]]
    capacities: np.ndarray
    commodities: list[Commodity]

    @classmethod
    def random(cls, n_nodes: int = 6, n_edges: int = 10,
               n_commodities: int = 3, seed: int = 42) -> MultiCommodityFlowInstance:
        """Generate a random multi-commodity flow instance.

        Args:
            n_nodes: Number of nodes.
            n_edges: Number of directed edges.
            n_commodities: Number of commodities.
            seed: Random seed for reproducibility.

        Returns:
            A random MultiCommodityFlowInstance.
        """
        rng = np.random.default_rng(seed)
        edges_set: set[tuple[int, int]] = set()
        while len(edges_set) < n_edges:
            u = rng.integers(0, n_nodes)
            v = rng.integers(0, n_nodes)
            if u != v:
                edges_set.add((int(u), int(v)))
        edges = sorted(edges_set)
        capacities = rng.uniform(5.0, 20.0, size=len(edges))

        commodities = []
        for _ in range(n_commodities):
            s = int(rng.integers(0, n_nodes))
            t = int(rng.integers(0, n_nodes))
            while t == s:
                t = int(rng.integers(0, n_nodes))
            demand = float(rng.uniform(1.0, 5.0))
            commodities.append(Commodity(source=s, sink=t, demand=demand))

        return cls(n_nodes=n_nodes, edges=edges,
                   capacities=capacities, commodities=commodities)


@dataclass
class MultiCommodityFlowSolution:
    """Solution to a multi-commodity flow problem.

    Attributes:
        flows: Dict mapping commodity index to dict of edge index -> flow value.
        total_flow: Total flow across all commodities.
        feasible: Whether the solution is feasible.
    """
    flows: dict[int, dict[int, float]]
    total_flow: float
    feasible: bool

    def __repr__(self) -> str:
        return (f"MultiCommodityFlowSolution(total_flow={self.total_flow:.2f}, "
                f"feasible={self.feasible}, n_commodities={len(self.flows)})")
