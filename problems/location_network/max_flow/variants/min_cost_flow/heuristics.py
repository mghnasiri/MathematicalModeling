"""
Minimum Cost Flow — Exact Algorithm.

Algorithms:
    - Successive Shortest Paths (SSP): Send flow along minimum-cost
      augmenting paths using Bellman-Ford on the residual graph.

Complexity: O(V^2 * E) with Bellman-Ford.

References:
    Ahuja, R.K., Magnanti, T.L. & Orlin, J.B. (1993). Network Flows:
    Theory, Algorithms, and Applications. Prentice Hall.
    ISBN 978-0136175490.
"""

from __future__ import annotations

import sys
import os
import importlib.util

import numpy as np

_this_dir = os.path.dirname(os.path.abspath(__file__))


def _load_mod(name, filepath):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_mod("mcf_instance_h", os.path.join(_this_dir, "instance.py"))
MinCostFlowInstance = _inst.MinCostFlowInstance
MinCostFlowSolution = _inst.MinCostFlowSolution


def successive_shortest_paths(instance: MinCostFlowInstance) -> MinCostFlowSolution:
    """Successive Shortest Paths algorithm for Min-Cost Flow.

    Repeatedly finds shortest (cheapest) augmenting path from a supply
    node to a demand node and sends flow along it.

    Args:
        instance: MinCostFlowInstance.

    Returns:
        MinCostFlowSolution.
    """
    n = instance.n
    INF = float("inf")

    # Build residual graph
    # For each edge (u,v,cap,cost), create forward arc and reverse arc
    # Adjacency: adj[u] = list of (v, cap, cost, rev_index)
    adj: list[list[list]] = [[] for _ in range(n)]

    for u, v, cap, cost in instance.edges:
        # Forward arc
        fwd_idx = len(adj[u])
        rev_idx = len(adj[v])
        adj[u].append([v, cap, cost, rev_idx])
        adj[v].append([u, 0, -cost, fwd_idx])  # reverse arc

    supply = instance.supply.copy()
    total_cost = 0.0
    total_flow = 0.0

    # While there are supply nodes with excess
    while True:
        # Find a supply node and a demand node
        src = -1
        snk = -1
        for i in range(n):
            if supply[i] > 1e-8:
                src = i
            if supply[i] < -1e-8:
                snk = i
        if src < 0 or snk < 0:
            break

        # Bellman-Ford from src
        dist = [INF] * n
        dist[src] = 0
        parent = [(-1, -1)] * n  # (node, arc_index)
        in_queue = [False] * n

        queue = [src]
        in_queue[src] = True

        while queue:
            u = queue.pop(0)
            in_queue[u] = False
            for idx, (v, cap, cost, _) in enumerate(adj[u]):
                if cap > 1e-8 and dist[u] + cost < dist[v] - 1e-10:
                    dist[v] = dist[u] + cost
                    parent[v] = (u, idx)
                    if not in_queue[v]:
                        queue.append(v)
                        in_queue[v] = True

        if dist[snk] >= INF:
            break  # No augmenting path

        # Find bottleneck flow
        flow = min(supply[src], -supply[snk])
        v = snk
        while v != src:
            u, idx = parent[v]
            flow = min(flow, adj[u][idx][1])
            v = u

        # Augment flow
        v = snk
        while v != src:
            u, idx = parent[v]
            adj[u][idx][1] -= flow  # reduce forward capacity
            rev_idx = adj[u][idx][3]
            adj[v][rev_idx][1] += flow  # increase reverse capacity
            total_cost += flow * adj[u][idx][2]
            v = u

        supply[src] -= flow
        supply[snk] += flow
        total_flow += flow

    # Extract flow values
    flow_dict: dict[tuple[int, int], float] = {}
    edge_idx = 0
    for u, v, cap, cost in instance.edges:
        # Flow = original_cap - residual_cap
        residual_cap = adj[u][edge_idx * 1][1] if edge_idx < len(adj[u]) else 0
        # Find the correct arc
        for arc in adj[u]:
            if arc[0] == v and arc[2] == cost:
                f = cap - arc[1]
                if f > 1e-8:
                    flow_dict[(u, v)] = f
                break

    # Recompute flow from residual graph properly
    flow_dict = {}
    edge_i = 0
    for u, v, cap, cost in instance.edges:
        # Find forward arc for this edge
        for arc in adj[u]:
            if arc[0] == v and abs(arc[2] - cost) < 1e-10:
                f = cap - arc[1]
                if f > 1e-8:
                    flow_dict[(u, v)] = flow_dict.get((u, v), 0.0) + f
                break

    return MinCostFlowSolution(flow=flow_dict, total_cost=total_cost,
                               total_flow=total_flow)


if __name__ == "__main__":
    from instance import small_mcf_4

    inst = small_mcf_4()
    sol = successive_shortest_paths(inst)
    print(f"SSP: {sol}")
    for (u, v), f in sol.flow.items():
        print(f"  ({u} -> {v}): {f:.1f}")
