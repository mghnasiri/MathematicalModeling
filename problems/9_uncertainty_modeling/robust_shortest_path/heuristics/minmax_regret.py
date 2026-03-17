"""
Min-Max Regret Heuristics for Robust Shortest Path

The regret of path P under scenario s is:
    regret_s(P) = cost_s(P) - cost_s(P*_s)
where P*_s is the optimal path under scenario s.

The min-max regret path minimizes: max_{s in S} regret_s(P).

Min-max regret is NP-hard for general interval data, but tractable
for discrete scenarios via enumeration of candidate paths.

References:
    - Averbakh, I. & Lebedev, V. (2004). Interval data minmax regret network
      optimization problems. DAM, 138(3), 289-301.
      https://doi.org/10.1016/S0166-218X(03)00462-1
    - Montemanni, R. & Gambardella, L.M. (2004). An exact algorithm for the
      robust shortest path problem with interval data. Computers & OR, 31(10),
      1667-1680. https://doi.org/10.1016/S0305-0548(03)00114-X
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


def _dijkstra_path(adj: dict[int, list[tuple[int, float]]],
                   source: int, target: int, n: int) -> tuple[float, list[int]]:
    """Dijkstra returning (distance, path)."""
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


def minmax_regret_enumeration(instance: RobustSPInstance) -> RobustSPSolution:
    """Solve min-max regret via scenario-wise shortest paths.

    1. For each scenario s, compute optimal path P*_s and cost c*_s.
    2. Evaluate each candidate path's regret across all scenarios.
    3. Return path with minimum max regret.

    Args:
        instance: RobustSPInstance.

    Returns:
        RobustSPSolution with min-max regret path.
    """
    S = instance.n_scenarios

    # Step 1: Compute optimal costs per scenario
    optimal_costs = []
    candidate_paths = []
    for s in range(S):
        adj = instance.adjacency_list(s)
        cost, path = _dijkstra_path(adj, instance.source, instance.target,
                                    instance.n_nodes)
        optimal_costs.append(cost)
        if path:
            candidate_paths.append(path)

    # Deduplicate candidates
    unique_paths = []
    seen = set()
    for p in candidate_paths:
        key = tuple(p)
        if key not in seen:
            seen.add(key)
            unique_paths.append(p)

    if not unique_paths:
        return RobustSPSolution(
            path=[], max_cost=float("inf"), expected_cost=float("inf"),
            max_regret=float("inf"),
        )

    # Step 2: Evaluate regret
    best_path = unique_paths[0]
    best_max_regret = float("inf")
    best_costs: list[float] = []

    for path in unique_paths:
        costs = [instance.path_cost(path, s) for s in range(S)]
        regrets = [costs[s] - optimal_costs[s] for s in range(S)]
        mr = max(regrets)
        if mr < best_max_regret:
            best_max_regret = mr
            best_path = path
            best_costs = costs

    return RobustSPSolution(
        path=best_path,
        max_cost=max(best_costs),
        expected_cost=instance.expected_cost(best_path),
        scenario_costs=best_costs,
        max_regret=best_max_regret,
    )


def midpoint_scenario(instance: RobustSPInstance) -> RobustSPSolution:
    """Heuristic: solve shortest path on the mean-weight scenario.

    Takes the element-wise mean of all scenario weights and finds
    the shortest path on this "average" graph.

    Args:
        instance: RobustSPInstance.

    Returns:
        RobustSPSolution using mean-weight path.
    """
    S = instance.n_scenarios
    mean_weights = np.mean(instance.weight_scenarios, axis=0)

    # Build adjacency with mean weights
    adj: dict[int, list[tuple[int, float]]] = {i: [] for i in range(instance.n_nodes)}
    for e_idx, (u, v) in enumerate(instance.edges):
        adj[u].append((v, mean_weights[e_idx]))

    _, path = _dijkstra_path(adj, instance.source, instance.target, instance.n_nodes)

    if not path:
        return RobustSPSolution(
            path=[], max_cost=float("inf"), expected_cost=float("inf"),
        )

    scenario_costs = [instance.path_cost(path, s) for s in range(S)]

    # Compute regret
    optimal_costs = []
    for s in range(S):
        adj_s = instance.adjacency_list(s)
        cost_s, _ = _dijkstra_path(adj_s, instance.source, instance.target,
                                   instance.n_nodes)
        optimal_costs.append(cost_s)

    regrets = [scenario_costs[s] - optimal_costs[s] for s in range(S)]

    return RobustSPSolution(
        path=path,
        max_cost=max(scenario_costs),
        expected_cost=instance.expected_cost(path),
        scenario_costs=scenario_costs,
        max_regret=max(regrets),
    )


if __name__ == "__main__":
    inst = RobustSPInstance.random(n_nodes=5, n_scenarios=4)
    sol_enum = minmax_regret_enumeration(inst)
    print(f"Min-max regret (enum): {sol_enum}")
    sol_mid = midpoint_scenario(inst)
    print(f"Midpoint heuristic: {sol_mid}")
