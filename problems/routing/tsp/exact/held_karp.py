"""
Held-Karp Algorithm — Exact TSP solver via dynamic programming.

Problem: TSP (Traveling Salesman Problem)
Complexity: O(2^n * n^2) time, O(2^n * n) space
Practical limit: n <= 20-23 cities

The Held-Karp algorithm uses bitmask dynamic programming to find the
optimal Hamiltonian cycle. dp[S][i] stores the minimum cost of a path
starting at city 0, visiting all cities in subset S, and ending at city i.

References:
    Held, M. & Karp, R.M. (1962). A dynamic programming approach to
    sequencing problems. Journal of the Society for Industrial and
    Applied Mathematics, 10(1), 196-210.
    https://doi.org/10.1137/0110015

    Bellman, R. (1962). Dynamic programming treatment of the travelling
    salesman problem. Journal of the ACM, 9(1), 61-63.
    https://doi.org/10.1145/321105.321111
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

_inst = _load_module("tsp_instance_hk", os.path.join(_parent_dir, "instance.py"))
TSPInstance = _inst.TSPInstance
TSPSolution = _inst.TSPSolution


def held_karp(instance: TSPInstance) -> TSPSolution:
    """Solve TSP exactly using the Held-Karp dynamic programming algorithm.

    Args:
        instance: A TSPInstance with n cities.

    Returns:
        TSPSolution with optimal tour and distance.

    Raises:
        ValueError: If n > 23 (memory/time impractical).
    """
    n = instance.n
    if n > 23:
        raise ValueError(f"Held-Karp impractical for n={n} > 23")

    if n == 1:
        return TSPSolution(tour=[0], distance=0.0)

    if n == 2:
        d = instance.distance_matrix[0][1] + instance.distance_matrix[1][0]
        return TSPSolution(tour=[0, 1], distance=d)

    dist = instance.distance_matrix
    full_mask = (1 << n) - 1

    # dp[S][i] = min cost path from 0, visiting all cities in S, ending at i
    # S is a bitmask; city 0 is always in S
    INF = float("inf")
    dp = np.full((1 << n, n), INF)
    parent = np.full((1 << n, n), -1, dtype=int)

    # Base case: start at city 0
    dp[1][0] = 0.0

    for S in range(1, 1 << n):
        if not (S & 1):
            # City 0 must always be in the subset
            continue
        for j in range(n):
            if not (S & (1 << j)):
                continue
            if dp[S][j] == INF:
                continue
            # Try extending to city k not in S
            for k in range(1, n):  # skip city 0 for intermediate
                if S & (1 << k):
                    continue
                new_S = S | (1 << k)
                new_cost = dp[S][j] + dist[j][k]
                if new_cost < dp[new_S][k]:
                    dp[new_S][k] = new_cost
                    parent[new_S][k] = j

    # Find optimal tour: close the cycle back to city 0
    best_cost = INF
    best_last = -1
    for j in range(1, n):
        cost = dp[full_mask][j] + dist[j][0]
        if cost < best_cost:
            best_cost = cost
            best_last = j

    # Reconstruct tour
    tour = []
    S = full_mask
    j = best_last
    while j != -1:
        tour.append(j)
        prev = parent[S][j]
        S = S ^ (1 << j)
        j = prev

    tour.reverse()

    return TSPSolution(tour=tour, distance=best_cost)


if __name__ == "__main__":
    from instance import small4, small5, gr17

    print("=== Held-Karp DP ===\n")

    s4 = small4()
    sol = held_karp(s4)
    print(f"small4: {sol}")

    s5 = small5()
    sol = held_karp(s5)
    print(f"small5: {sol}")

    g17 = gr17()
    sol = held_karp(g17)
    print(f"gr17:   {sol}")
    print(f"  (optimal = 2085)")
