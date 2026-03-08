"""
Dijkstra's Algorithm — Shortest path in graphs with non-negative weights.

Problem: Single-Source Shortest Path (SSSP)
Complexity: O((V + E) log V) with binary heap

Computes shortest paths from a source node to all other nodes (or a
specific target) in a graph with non-negative edge weights. Uses a
priority queue (min-heap) for efficient minimum distance extraction.

References:
    Dijkstra, E.W. (1959). A note on two problems in connexion with
    graphs. Numerische Mathematik, 1(1), 269-271.
    https://doi.org/10.1007/BF01386390

    Fredman, M.L. & Tarjan, R.E. (1987). Fibonacci heaps and their
    uses in improved network optimization algorithms. Journal of the
    ACM, 34(3), 596-615.
    https://doi.org/10.1145/28869.28874
"""

from __future__ import annotations

import os
import sys
import heapq
import importlib.util

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_mod(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_mod("sp_instance_dj", os.path.join(_parent_dir, "instance.py"))
ShortestPathInstance = _inst.ShortestPathInstance
ShortestPathSolution = _inst.ShortestPathSolution


def dijkstra(
    instance: ShortestPathInstance,
    source: int,
    target: int,
) -> ShortestPathSolution:
    """Find shortest path from source to target using Dijkstra's algorithm.

    Args:
        instance: A ShortestPathInstance (non-negative weights).
        source: Source node.
        target: Target node.

    Returns:
        ShortestPathSolution. If no path exists, distance = inf, path = [].

    Raises:
        ValueError: If graph has negative edge weights.
    """
    if instance.has_negative_weights():
        raise ValueError("Dijkstra requires non-negative edge weights")

    n = instance.n
    dist = [float("inf")] * n
    prev = [-1] * n
    dist[source] = 0.0

    # Priority queue: (distance, node)
    pq = [(0.0, source)]
    visited = [False] * n

    while pq:
        d, u = heapq.heappop(pq)
        if visited[u]:
            continue
        visited[u] = True

        if u == target:
            break

        for v, w in instance.adjacency[u]:
            new_dist = d + w
            if new_dist < dist[v]:
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(pq, (new_dist, v))

    if dist[target] == float("inf"):
        return ShortestPathSolution(
            source=source, target=target,
            path=[], distance=float("inf"),
        )

    # Reconstruct path
    path = []
    node = target
    while node != -1:
        path.append(node)
        node = prev[node]
    path.reverse()

    return ShortestPathSolution(
        source=source, target=target,
        path=path, distance=dist[target],
    )


def dijkstra_all(
    instance: ShortestPathInstance, source: int
) -> tuple[list[float], list[int]]:
    """Compute shortest distances from source to all nodes.

    Args:
        instance: A ShortestPathInstance (non-negative weights).
        source: Source node.

    Returns:
        Tuple of (distances, predecessors).
    """
    n = instance.n
    dist = [float("inf")] * n
    prev = [-1] * n
    dist[source] = 0.0

    pq = [(0.0, source)]
    visited = [False] * n

    while pq:
        d, u = heapq.heappop(pq)
        if visited[u]:
            continue
        visited[u] = True

        for v, w in instance.adjacency[u]:
            new_dist = d + w
            if new_dist < dist[v]:
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(pq, (new_dist, v))

    return dist, prev


if __name__ == "__main__":
    from instance import simple_graph_5

    inst = simple_graph_5()
    sol = dijkstra(inst, 0, 4)
    print(f"Dijkstra 0->4: {sol}")

    dists, _ = dijkstra_all(inst, 0)
    print(f"All distances from 0: {dists}")
