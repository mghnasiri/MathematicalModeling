"""
All-Pairs Shortest Path — Exact Algorithms.

Algorithms:
    - Floyd-Warshall: O(V^3), handles negative weights, no negative cycles.
    - Repeated Dijkstra: O(V * (V + E) log V), non-negative weights only.

References:
    Floyd, R.W. (1962). Algorithm 97: Shortest path. Communications
    of the ACM, 5(6), 345. https://doi.org/10.1145/367766.368168

    Dijkstra, E.W. (1959). A note on two problems in connexion with
    graphs. Numerische Mathematik, 1(1), 269-271.
    https://doi.org/10.1007/BF01386390
"""

from __future__ import annotations

import sys
import os
import importlib.util
import heapq

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


_inst = _load_mod("apsp_instance_h", os.path.join(_this_dir, "instance.py"))
APSPInstance = _inst.APSPInstance
APSPSolution = _inst.APSPSolution


def floyd_warshall(instance: APSPInstance) -> APSPSolution:
    """Floyd-Warshall algorithm for APSP.

    Computes shortest paths between all pairs in O(V^3).
    Handles negative weights but assumes no negative cycles.

    Args:
        instance: APSP instance.

    Returns:
        APSPSolution with distance and next-hop matrices.
    """
    n = instance.n
    dist = instance.weight_matrix.copy()
    next_hop = np.full((n, n), -1, dtype=int)

    # Initialize next-hop matrix
    for i in range(n):
        for j in range(n):
            if i != j and dist[i][j] < np.inf:
                next_hop[i][j] = j

    # DP relaxation
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j] - 1e-10:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    next_hop[i][j] = next_hop[i][k]

    return APSPSolution(dist_matrix=dist, next_matrix=next_hop)


def repeated_dijkstra(instance: APSPInstance) -> APSPSolution:
    """Repeated Dijkstra for APSP.

    Runs Dijkstra from each source. Requires non-negative weights.
    O(V * (V + E) log V).

    Args:
        instance: APSP instance (must have non-negative weights).

    Returns:
        APSPSolution.
    """
    n = instance.n
    dist = np.full((n, n), np.inf)
    next_hop = np.full((n, n), -1, dtype=int)

    for s in range(n):
        dist[s][s] = 0.0
        # Build adjacency from weight matrix
        pq = [(0.0, s)]
        visited = set()
        prev = [-1] * n

        while pq:
            d, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)
            dist[s][u] = d

            for v in range(n):
                if v == u:
                    continue
                w = instance.weight_matrix[u][v]
                if w < np.inf and d + w < dist[s][v]:
                    dist[s][v] = d + w
                    prev[v] = u
                    heapq.heappush(pq, (d + w, v))

        # Build next-hop from prev
        for t in range(n):
            if t == s:
                continue
            if dist[s][t] < np.inf:
                # Trace back from t to s
                node = t
                while prev[node] != s and prev[node] != -1:
                    node = prev[node]
                if prev[node] == s:
                    next_hop[s][t] = node
                elif node == t and prev[t] == s:
                    next_hop[s][t] = t

    return APSPSolution(dist_matrix=dist, next_matrix=next_hop)


if __name__ == "__main__":
    from instance import small_apsp_4

    inst = small_apsp_4()
    sol1 = floyd_warshall(inst)
    print(f"Floyd-Warshall: {sol1}")
    print(f"  Distances:\n{sol1.dist_matrix}")

    sol2 = repeated_dijkstra(inst)
    print(f"Repeated Dijkstra: {sol2}")
