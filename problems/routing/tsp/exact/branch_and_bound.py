"""
Branch and Bound — Exact TSP solver with 1-tree lower bound.

Problem: TSP (Traveling Salesman Problem)
Complexity: O(n! worst case), practical for n <= ~25 with good bounds

Uses a depth-first search with a minimum spanning tree (1-tree) based
lower bound. The upper bound is initialized with the nearest neighbor
heuristic for effective pruning.

References:
    Little, J.D.C., Murty, K.G., Sweeney, D.W. & Karel, C. (1963).
    An algorithm for the traveling salesman problem.
    Operations Research, 11(6), 972-989.
    https://doi.org/10.1287/opre.11.6.972

    Held, M. & Karp, R.M. (1970). The traveling-salesman problem and
    minimum spanning trees. Operations Research, 18(6), 1138-1162.
    https://doi.org/10.1287/opre.18.6.1138
"""

from __future__ import annotations

import os
import numpy as np

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _load_module(name, filepath):
    import importlib.util
    import sys as _sys
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    _sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_inst = _load_module("tsp_instance_bnb", os.path.join(_parent_dir, "instance.py"))
TSPInstance = _inst.TSPInstance
TSPSolution = _inst.TSPSolution


def _mst_cost(dist: np.ndarray, nodes: list[int]) -> float:
    """Compute MST cost on a subset of nodes using Prim's algorithm.

    Args:
        dist: Full distance matrix.
        nodes: List of node indices to include.

    Returns:
        Total MST edge weight.
    """
    if len(nodes) <= 1:
        return 0.0

    n = len(nodes)
    in_tree = [False] * n
    min_edge = [float("inf")] * n
    min_edge[0] = 0.0
    total = 0.0

    for _ in range(n):
        # Find cheapest node not in tree
        u = -1
        for i in range(n):
            if not in_tree[i] and (u == -1 or min_edge[i] < min_edge[u]):
                u = i
        in_tree[u] = True
        total += min_edge[u]

        # Update min edges
        for v in range(n):
            if not in_tree[v]:
                d = dist[nodes[u]][nodes[v]]
                if d < min_edge[v]:
                    min_edge[v] = d

    return total


def _one_tree_lower_bound(
    dist: np.ndarray, n: int, partial_tour: list[int], visited: set[int]
) -> float:
    """Compute a 1-tree lower bound for the partial tour.

    The bound includes:
    - Cost of edges already in the partial tour
    - MST on the unvisited nodes plus the last visited node
    - Cheapest edge connecting the first node to unvisited nodes

    Args:
        dist: Distance matrix.
        n: Total number of cities.
        partial_tour: Current partial tour.
        visited: Set of visited cities.

    Returns:
        Lower bound on the complete tour cost.
    """
    if len(partial_tour) <= 1:
        return _mst_cost(dist, list(range(n)))

    # Cost of partial tour so far
    tour_cost = 0.0
    for i in range(len(partial_tour) - 1):
        tour_cost += dist[partial_tour[i]][partial_tour[i + 1]]

    unvisited = [c for c in range(n) if c not in visited]
    if not unvisited:
        # Complete tour — add return edge
        return tour_cost + dist[partial_tour[-1]][partial_tour[0]]

    # MST on unvisited nodes + endpoints
    remaining = unvisited + [partial_tour[-1]]
    mst_cost = _mst_cost(dist, remaining)

    # Cheapest connection from start to unvisited
    min_start = min(dist[partial_tour[0]][u] for u in unvisited)

    return tour_cost + mst_cost + min_start


def _nearest_neighbor_tour(instance: TSPInstance) -> tuple[list[int], float]:
    """Quick nearest neighbor heuristic for initial upper bound.

    Args:
        instance: TSP instance.

    Returns:
        Tuple of (tour, distance).
    """
    n = instance.n
    dist = instance.distance_matrix
    visited = {0}
    tour = [0]

    for _ in range(n - 1):
        current = tour[-1]
        best_next = -1
        best_dist = float("inf")
        for j in range(n):
            if j not in visited and dist[current][j] < best_dist:
                best_dist = dist[current][j]
                best_next = j
        tour.append(best_next)
        visited.add(best_next)

    return tour, instance.tour_distance(tour)


def branch_and_bound(
    instance: TSPInstance, time_limit: float | None = None
) -> TSPSolution:
    """Solve TSP exactly using Branch and Bound with 1-tree lower bound.

    Args:
        instance: A TSPInstance.
        time_limit: Optional time limit in seconds (not enforced precisely).

    Returns:
        TSPSolution with optimal tour and distance.
    """
    import time

    n = instance.n
    dist = instance.distance_matrix

    if n == 1:
        return TSPSolution(tour=[0], distance=0.0)
    if n == 2:
        d = dist[0][1] + dist[1][0]
        return TSPSolution(tour=[0, 1], distance=d)

    # Initialize with nearest neighbor heuristic
    best_tour, best_cost = _nearest_neighbor_tour(instance)
    start_time = time.time()

    # DFS stack: (partial_tour, visited_set)
    stack: list[tuple[list[int], set[int]]] = [([0], {0})]

    while stack:
        if time_limit and (time.time() - start_time) > time_limit:
            break

        partial_tour, visited = stack.pop()
        current = partial_tour[-1]

        if len(partial_tour) == n:
            # Complete tour — check if better
            cost = 0.0
            for i in range(n):
                cost += dist[partial_tour[i]][partial_tour[(i + 1) % n]]
            if cost < best_cost:
                best_cost = cost
                best_tour = partial_tour[:]
            continue

        # Branch: try extending with each unvisited city
        candidates = []
        for j in range(n):
            if j in visited:
                continue
            new_tour = partial_tour + [j]
            new_visited = visited | {j}
            lb = _one_tree_lower_bound(dist, n, new_tour, new_visited)
            if lb < best_cost:
                candidates.append((lb, j, new_tour, new_visited))

        # Sort by lower bound (best-first within DFS level)
        candidates.sort(key=lambda x: -x[0])  # reverse for stack (LIFO)
        for _, _, new_tour, new_visited in candidates:
            stack.append((new_tour, new_visited))

    return TSPSolution(tour=best_tour, distance=best_cost)


if __name__ == "__main__":
    from instance import small4, small5, gr17

    print("=== Branch and Bound ===\n")

    s4 = small4()
    sol = branch_and_bound(s4)
    print(f"small4: {sol}")

    s5 = small5()
    sol = branch_and_bound(s5)
    print(f"small5: {sol}")

    g17 = gr17()
    sol = branch_and_bound(g17, time_limit=30)
    print(f"gr17:   {sol}")
    print(f"  (optimal = 2085)")
