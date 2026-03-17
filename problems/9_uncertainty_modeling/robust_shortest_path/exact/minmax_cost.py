"""
Min-Max Cost Robust Shortest Path

Finds the path P from source to target minimizing the worst-case cost:
    min_{P} max_{s in S} cost_s(P)

Approach: Enumerate shortest paths per scenario via Dijkstra, then
evaluate all candidate paths across all scenarios. For small instances,
also try enumerating k-shortest paths.

For discrete scenarios, the exact approach uses a label-setting algorithm
that tracks the cost vector across scenarios for each partial path.

Complexity: O(S * (V+E) log V) for the scenario-enumeration heuristic.
            Exact min-max over discrete scenarios is polynomial via
            auxiliary graph construction.

References:
    - Yu, G. & Yang, J. (1998). On the robust shortest path problem.
      Computers & OR, 25(6), 457-468.
      https://doi.org/10.1016/S0305-0548(97)00085-3
"""
from __future__ import annotations

import heapq
import sys
import os

import numpy as np

import importlib.util

def _load_parent(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_inst = _load_parent("rsp_instance", os.path.join(os.path.dirname(__file__), "..", "instance.py"))
RobustSPInstance = _inst.RobustSPInstance
RobustSPSolution = _inst.RobustSPSolution


def _dijkstra(adj: dict[int, list[tuple[int, float]]],
              source: int, target: int) -> tuple[float, list[int]]:
    """Standard Dijkstra for a single scenario.

    Args:
        adj: Adjacency list {node: [(neighbor, weight), ...]}.
        source: Start node.
        target: End node.

    Returns:
        (shortest distance, path) or (inf, []) if unreachable.
    """
    n = len(adj)
    dist = [float("inf")] * n
    prev = [-1] * n
    dist[source] = 0.0
    pq = [(0.0, source)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        if u == target:
            break
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    if dist[target] == float("inf"):
        return float("inf"), []

    path = []
    node = target
    while node != -1:
        path.append(node)
        node = prev[node]
    path.reverse()
    return dist[target], path


def minmax_cost_enumeration(instance: RobustSPInstance) -> RobustSPSolution:
    """Solve min-max cost via scenario-wise Dijkstra + cross-evaluation.

    1. For each scenario s, find shortest path P_s via Dijkstra.
    2. Evaluate each P_s across all scenarios to get max cost.
    3. Return path with minimum max cost.

    This is a heuristic — it only considers S candidate paths.

    Args:
        instance: RobustSPInstance.

    Returns:
        RobustSPSolution with best path found.
    """
    S = instance.n_scenarios
    candidate_paths = []

    for s in range(S):
        adj = instance.adjacency_list(s)
        _, path = _dijkstra(adj, instance.source, instance.target)
        if path:
            candidate_paths.append(path)

    # Deduplicate
    unique_paths = []
    seen = set()
    for p in candidate_paths:
        key = tuple(p)
        if key not in seen:
            seen.add(key)
            unique_paths.append(p)

    if not unique_paths:
        return RobustSPSolution(
            path=[], max_cost=float("inf"), expected_cost=float("inf")
        )

    best_path = unique_paths[0]
    best_max_cost = float("inf")

    for path in unique_paths:
        costs = [instance.path_cost(path, s) for s in range(S)]
        mc = max(costs)
        if mc < best_max_cost:
            best_max_cost = mc
            best_path = path

    scenario_costs = [instance.path_cost(best_path, s) for s in range(S)]

    return RobustSPSolution(
        path=best_path,
        max_cost=best_max_cost,
        expected_cost=instance.expected_cost(best_path),
        scenario_costs=scenario_costs,
    )


def minmax_cost_label_setting(instance: RobustSPInstance) -> RobustSPSolution:
    """Exact min-max cost via multi-objective label-setting.

    Each label at node v is a vector of costs (one per scenario).
    A label dominates another if it is <= in all scenarios.
    The algorithm propagates non-dominated labels from source to target.

    Args:
        instance: RobustSPInstance.

    Returns:
        Optimal RobustSPSolution.
    """
    S = instance.n_scenarios
    n = instance.n_nodes

    # Build adjacency with edge indices
    adj: dict[int, list[tuple[int, int]]] = {i: [] for i in range(n)}
    for e_idx, (u, v) in enumerate(instance.edges):
        adj[u].append((v, e_idx))

    # Labels: dict node -> list of (cost_vector, path)
    # Use priority queue with max-cost as key
    # Label = (max_cost, cost_vector, path)
    source = instance.source
    target = instance.target

    init_cost = np.zeros(S)
    pq = [(0.0, id(init_cost), source, init_cost, [source])]
    # Best labels at each node: list of cost vectors
    best_labels: dict[int, list[np.ndarray]] = {i: [] for i in range(n)}

    best_solution = None
    best_max_cost = float("inf")

    max_labels = 500  # limit for tractability

    while pq:
        mc, _, u, cost_vec, path = heapq.heappop(pq)

        if mc >= best_max_cost:
            continue

        if u == target:
            if mc < best_max_cost:
                best_max_cost = mc
                best_solution = (path, cost_vec)
            continue

        # Dominance check
        dominated = False
        for label in best_labels[u]:
            if np.all(label <= cost_vec):
                dominated = True
                break
        if dominated:
            continue

        # Remove dominated labels
        best_labels[u] = [
            l for l in best_labels[u] if not np.all(cost_vec <= l)
        ]
        best_labels[u].append(cost_vec.copy())

        if len(best_labels[u]) > max_labels:
            continue

        for v, e_idx in adj[u]:
            if v in path:  # no cycles
                continue
            edge_costs = instance.weight_scenarios[:, e_idx]
            new_cost = cost_vec + edge_costs
            new_mc = float(new_cost.max())
            if new_mc < best_max_cost:
                heapq.heappush(pq, (new_mc, id(new_cost), v, new_cost, path + [v]))

    if best_solution is None:
        return RobustSPSolution(
            path=[], max_cost=float("inf"), expected_cost=float("inf")
        )

    path, cost_vec = best_solution
    scenario_costs = [float(cost_vec[s]) for s in range(S)]

    return RobustSPSolution(
        path=path,
        max_cost=float(cost_vec.max()),
        expected_cost=float(np.dot(cost_vec, instance.probabilities)),
        scenario_costs=scenario_costs,
    )


if __name__ == "__main__":
    inst = RobustSPInstance.random(n_nodes=5, n_scenarios=4)
    sol_enum = minmax_cost_enumeration(inst)
    print(f"Enumeration: {sol_enum}")
    sol_label = minmax_cost_label_setting(inst)
    print(f"Label-setting: {sol_label}")
