"""Capacitated Arc Routing Problem (CARP) instance and solution definitions.

Problem: Given an undirected graph with required edges (each having a demand),
a depot node, and vehicles with capacity Q, find a set of routes starting
and ending at the depot that serve all required edges while respecting
vehicle capacity.

Notation: CARP
Complexity: NP-hard

References:
    Golden, B. L., & Wong, R. T. (1981). Capacitated arc routing problems.
    Networks, 11(3), 305-315.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class CARPInstance:
    """Capacitated Arc Routing Problem instance.

    Attributes:
        n_nodes: Number of nodes.
        edges: List of (u, v) tuples for all edges (undirected).
        costs: Cost (distance) for traversing each edge.
        demands: Demand on each edge (0 if not required).
        depot: Depot node index.
        capacity: Vehicle capacity.
    """
    n_nodes: int
    edges: list[tuple[int, int]]
    costs: np.ndarray
    demands: np.ndarray
    depot: int
    capacity: float

    @property
    def required_edges(self) -> list[int]:
        """Indices of edges with positive demand."""
        return [i for i, d in enumerate(self.demands) if d > 0]

    @property
    def n_required(self) -> int:
        """Number of required edges."""
        return len(self.required_edges)

    def shortest_paths(self) -> np.ndarray:
        """Compute all-pairs shortest path distances (Floyd-Warshall).

        Returns:
            n_nodes x n_nodes distance matrix.
        """
        dist = np.full((self.n_nodes, self.n_nodes), np.inf)
        np.fill_diagonal(dist, 0.0)
        for idx, (u, v) in enumerate(self.edges):
            c = self.costs[idx]
            if c < dist[u][v]:
                dist[u][v] = c
                dist[v][u] = c
        for k in range(self.n_nodes):
            for i in range(self.n_nodes):
                for j in range(self.n_nodes):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        return dist

    @classmethod
    def random(cls, n_nodes: int = 8, n_edges: int = 14,
               n_required: int = 8, capacity: float = 20.0,
               seed: int = 42) -> CARPInstance:
        """Generate a random CARP instance.

        Args:
            n_nodes: Number of nodes.
            n_edges: Number of edges.
            n_required: Number of required edges.
            capacity: Vehicle capacity.
            seed: Random seed.

        Returns:
            A random CARPInstance.
        """
        rng = np.random.default_rng(seed)

        # Build a connected graph: first a spanning tree
        edges_set: set[tuple[int, int]] = set()
        nodes_in = {0}
        nodes_out = set(range(1, n_nodes))
        while nodes_out:
            u = rng.choice(list(nodes_in))
            v = rng.choice(list(nodes_out))
            edge = (min(u, v), max(u, v))
            edges_set.add(edge)
            nodes_in.add(v)
            nodes_out.remove(v)

        # Add more edges
        while len(edges_set) < n_edges:
            u = int(rng.integers(0, n_nodes))
            v = int(rng.integers(0, n_nodes))
            if u != v:
                edge = (min(u, v), max(u, v))
                edges_set.add(edge)

        edges = sorted(edges_set)
        costs = rng.uniform(1.0, 10.0, size=len(edges))
        demands = np.zeros(len(edges))

        # Mark some edges as required
        req_count = min(n_required, len(edges))
        req_indices = rng.choice(len(edges), size=req_count, replace=False)
        for idx in req_indices:
            demands[idx] = float(rng.uniform(1.0, 5.0))

        return cls(n_nodes=n_nodes, edges=edges, costs=costs,
                   demands=demands, depot=0, capacity=capacity)


@dataclass
class CARPSolution:
    """Solution to a CARP instance.

    Attributes:
        routes: List of routes; each route is a list of (edge_index, direction)
                where direction is (u,v) indicating traversal order.
        total_cost: Total traversal cost including deadheading.
        feasible: Whether capacity constraints are satisfied and all required
                  edges are served.
    """
    routes: list[list[tuple[int, tuple[int, int]]]]
    total_cost: float
    feasible: bool

    def __repr__(self) -> str:
        return (f"CARPSolution(total_cost={self.total_cost:.2f}, "
                f"n_routes={len(self.routes)}, feasible={self.feasible})")
