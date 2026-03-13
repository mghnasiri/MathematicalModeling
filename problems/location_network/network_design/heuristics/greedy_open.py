"""Greedy heuristic for Fixed-Charge Network Design.

Algorithm: Iteratively open the edge that provides the best cost-to-benefit
ratio for satisfying remaining unsatisfied demand. Routes flow via shortest
paths on opened edges.

Complexity: O(E^2 * V) in the worst case.

References:
    Magnanti, T. L., & Wong, R. T. (1984). Network design and transportation
    planning: Models and algorithms. Transportation Science, 18(1), 1-55.
"""
import sys
import os
import importlib.util
from collections import deque

import numpy as np

def _load_parent(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_inst = _load_parent(
    "nd_instance",
    os.path.join(os.path.dirname(__file__), "..", "instance.py")
)

NetworkDesignInstance = _inst.NetworkDesignInstance
NetworkDesignSolution = _inst.NetworkDesignSolution


def _can_route(instance: NetworkDesignInstance, open_edges: set[int]) -> tuple[bool, dict[int, float]]:
    """Check if all demands can be routed using open edges.

    Uses a simple max-flow style approach: for each supply-demand pair,
    try to find a path and push flow.

    Returns:
        (feasible, flows_dict)
    """
    n = instance.n_nodes
    # Build adjacency from open edges with remaining capacity
    cap = {}
    for e_idx in open_edges:
        u, v = instance.potential_edges[e_idx]
        cap[(u, v, e_idx)] = instance.edge_capacities[e_idx]

    # Find supply and demand nodes
    supplies = {}
    demands_map = {}
    for i in range(n):
        if instance.demands[i] < 0:
            supplies[i] = -instance.demands[i]
        elif instance.demands[i] > 0:
            demands_map[i] = instance.demands[i]

    flows: dict[int, float] = {e: 0.0 for e in open_edges}

    # For each demand node, try to route from any supply node
    remaining_supply = dict(supplies)
    remaining_demand = dict(demands_map)

    for sink, dem in sorted(demands_map.items()):
        needed = dem
        for source in list(remaining_supply.keys()):
            if needed <= 1e-9:
                break
            available = remaining_supply[source]
            if available <= 1e-9:
                continue
            send = min(needed, available)

            # BFS to find path from source to sink on open edges with capacity
            path = _bfs_path(instance, open_edges, flows, source, sink)
            if path is None:
                continue

            # Find bottleneck
            bottleneck = send
            for e_idx in path:
                u, v = instance.potential_edges[e_idx]
                remaining_cap = instance.edge_capacities[e_idx] - flows[e_idx]
                bottleneck = min(bottleneck, remaining_cap)

            if bottleneck <= 1e-9:
                continue

            actual_send = min(send, bottleneck)
            for e_idx in path:
                flows[e_idx] += actual_send
            needed -= actual_send
            remaining_supply[source] -= actual_send

        remaining_demand[sink] = needed

    feasible = all(v <= 1e-9 for v in remaining_demand.values())
    return feasible, flows


def _bfs_path(instance: NetworkDesignInstance, open_edges: set[int],
              flows: dict[int, float], source: int, sink: int) -> list[int] | None:
    """BFS to find a path from source to sink using open edges with remaining capacity."""
    n = instance.n_nodes
    # Build adjacency
    adj: dict[int, list[tuple[int, int]]] = {i: [] for i in range(n)}
    for e_idx in open_edges:
        u, v = instance.potential_edges[e_idx]
        remaining = instance.edge_capacities[e_idx] - flows.get(e_idx, 0.0)
        if remaining > 1e-9:
            adj[u].append((v, e_idx))

    visited = {source}
    queue = deque([(source, [])])
    while queue:
        node, path = queue.popleft()
        if node == sink:
            return path
        for neighbor, e_idx in adj[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [e_idx]))
    return None


def greedy_open(instance: NetworkDesignInstance) -> NetworkDesignSolution:
    """Greedy edge-opening heuristic for network design.

    Iteratively opens the edge with the best cost-effectiveness
    (lowest fixed cost per unit of new flow enabled).

    Args:
        instance: A NetworkDesignInstance.

    Returns:
        NetworkDesignSolution.
    """
    n_edges = len(instance.potential_edges)
    open_edges: set[int] = set()
    closed_edges = set(range(n_edges))

    # Check if there is any demand to satisfy
    has_demand = any(abs(d) > 1e-9 for d in instance.demands)
    if not has_demand:
        return NetworkDesignSolution(
            open_edges=set(), flows={}, total_cost=0.0,
            fixed_cost=0.0, variable_cost=0.0, feasible=True
        )

    # Greedy: open edges one by one, cheapest first by fixed cost
    sorted_edges = sorted(closed_edges, key=lambda e: instance.fixed_costs[e])

    for e_idx in sorted_edges:
        open_edges.add(e_idx)
        feasible, flows = _can_route(instance, open_edges)
        if feasible:
            break
    else:
        # Opened all edges, check feasibility
        feasible, flows = _can_route(instance, open_edges)

    # Compute costs
    fixed_cost = sum(instance.fixed_costs[e] for e in open_edges)
    variable_cost = sum(
        flows.get(e, 0.0) * instance.unit_costs[e] for e in open_edges
    )
    total_cost = fixed_cost + variable_cost

    return NetworkDesignSolution(
        open_edges=open_edges, flows=flows, total_cost=total_cost,
        fixed_cost=fixed_cost, variable_cost=variable_cost, feasible=feasible
    )


if __name__ == "__main__":
    inst = NetworkDesignInstance.random()
    sol = greedy_open(inst)
    print(f"Instance: {inst.n_nodes} nodes, {len(inst.potential_edges)} edges")
    print(sol)
