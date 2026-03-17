"""
Greedy (Nearest Edge) Heuristic — TSP construction by adding shortest edges.

Problem: TSP (Traveling Salesman Problem)
Complexity: O(n^2 log n) — dominated by edge sorting

Sort all edges by distance. Add each edge to the tour if it doesn't
create a vertex with degree > 2 or form a premature cycle (cycle of
length < n). The result is a Hamiltonian cycle.

Approximation ratio: O(log n) worst case for metric TSP.

References:
    Bentley, J.L. (1992). Fast algorithms for geometric traveling
    salesman problems. ORSA Journal on Computing, 4(4), 387-411.
    https://doi.org/10.1287/ijoc.4.4.387
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

_inst = _load_module("tsp_instance_gr", os.path.join(_parent_dir, "instance.py"))
TSPInstance = _inst.TSPInstance
TSPSolution = _inst.TSPSolution


def greedy(instance: TSPInstance) -> TSPSolution:
    """Construct a tour using the greedy (nearest edge) heuristic.

    Args:
        instance: A TSPInstance.

    Returns:
        TSPSolution with the constructed tour.
    """
    n = instance.n
    dist = instance.distance_matrix

    if n <= 2:
        tour = list(range(n))
        return TSPSolution(tour=tour, distance=instance.tour_distance(tour))

    # Generate and sort all edges by distance
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((dist[i][j], i, j))
    edges.sort()

    # Build tour by adding shortest feasible edges
    degree = [0] * n
    adj = [[] for _ in range(n)]
    edges_added = 0

    # Union-Find for cycle detection
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        parent[px] = py

    for d, i, j in edges:
        if edges_added == n:
            break
        # Skip if either vertex already has degree 2
        if degree[i] >= 2 or degree[j] >= 2:
            continue
        # Skip if adding edge creates a premature cycle
        if edges_added < n - 1 and find(i) == find(j):
            continue

        adj[i].append(j)
        adj[j].append(i)
        degree[i] += 1
        degree[j] += 1
        union(i, j)
        edges_added += 1

    # Extract tour from adjacency list
    tour = [0]
    visited = {0}
    current = 0
    while len(tour) < n:
        for neighbor in adj[current]:
            if neighbor not in visited:
                tour.append(neighbor)
                visited.add(neighbor)
                current = neighbor
                break

    return TSPSolution(tour=tour, distance=instance.tour_distance(tour))


if __name__ == "__main__":
    from instance import small4, small5, gr17

    print("=== Greedy Heuristic ===\n")

    for name, inst_fn in [("small4", small4), ("small5", small5), ("gr17", gr17)]:
        inst = inst_fn()
        sol = greedy(inst)
        print(f"{name}: distance={sol.distance:.1f}, tour={sol.tour}")
