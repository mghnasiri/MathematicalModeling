"""Fixed-Charge Network Design Problem instance and solution definitions.

Problem: Given a set of potential edges with fixed opening costs and per-unit
flow costs, and nodes with supply/demand, select edges to open and route flows
to satisfy all demands at minimum total cost (fixed + variable).

Notation: FCNDP
Complexity: NP-hard

References:
    Magnanti, T. L., & Wong, R. T. (1984). Network design and transportation
    planning: Models and algorithms. Transportation Science, 18(1), 1-55.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class NetworkDesignInstance:
    """Fixed-charge network design problem instance.

    Attributes:
        n_nodes: Number of nodes.
        potential_edges: List of (from, to) tuples for potential edges.
        fixed_costs: Fixed cost for opening each edge.
        unit_costs: Per-unit flow cost on each edge.
        edge_capacities: Capacity of each edge if opened.
        demands: Demand at each node (positive = demand, negative = supply).
    """
    n_nodes: int
    potential_edges: list[tuple[int, int]]
    fixed_costs: np.ndarray
    unit_costs: np.ndarray
    edge_capacities: np.ndarray
    demands: np.ndarray

    @classmethod
    def random(cls, n_nodes: int = 6, n_edges: int = 12,
               seed: int = 42) -> NetworkDesignInstance:
        """Generate a random network design instance.

        Args:
            n_nodes: Number of nodes.
            n_edges: Number of potential edges.
            seed: Random seed.

        Returns:
            A random NetworkDesignInstance.
        """
        rng = np.random.default_rng(seed)
        edges_set: set[tuple[int, int]] = set()
        while len(edges_set) < n_edges:
            u = rng.integers(0, n_nodes)
            v = rng.integers(0, n_nodes)
            if u != v:
                edges_set.add((int(u), int(v)))
        edges = sorted(edges_set)

        fixed_costs = rng.uniform(10.0, 50.0, size=len(edges))
        unit_costs = rng.uniform(1.0, 5.0, size=len(edges))
        edge_capacities = rng.uniform(10.0, 30.0, size=len(edges))

        # Create balanced demands: first node is source, last is sink
        demands = np.zeros(n_nodes)
        demands[0] = -10.0  # supply
        demands[-1] = 10.0  # demand

        return cls(n_nodes=n_nodes, potential_edges=edges,
                   fixed_costs=fixed_costs, unit_costs=unit_costs,
                   edge_capacities=edge_capacities, demands=demands)


@dataclass
class NetworkDesignSolution:
    """Solution to a network design problem.

    Attributes:
        open_edges: Set of edge indices that are opened.
        flows: Dict mapping edge index to flow value.
        total_cost: Total cost (fixed + variable).
        fixed_cost: Total fixed cost component.
        variable_cost: Total variable cost component.
        feasible: Whether all demands are satisfied.
    """
    open_edges: set[int]
    flows: dict[int, float]
    total_cost: float
    fixed_cost: float
    variable_cost: float
    feasible: bool

    def __repr__(self) -> str:
        return (f"NetworkDesignSolution(total_cost={self.total_cost:.2f}, "
                f"open_edges={len(self.open_edges)}, feasible={self.feasible})")
