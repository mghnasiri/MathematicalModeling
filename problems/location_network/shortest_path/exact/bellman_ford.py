"""
Bellman-Ford Algorithm — Shortest path with negative edge weights.

Problem: Single-Source Shortest Path (SSSP)
Complexity: O(V * E)

Handles negative edge weights and detects negative-weight cycles.
Relaxes all edges V-1 times; if further relaxation is possible,
a negative cycle exists.

References:
    Bellman, R. (1958). On a routing problem. Quarterly of Applied
    Mathematics, 16(1), 87-90.
    https://doi.org/10.1090/qam/102435

    Ford, L.R. (1956). Network flow theory. Paper P-923, RAND
    Corporation, Santa Monica, CA.
"""

from __future__ import annotations

import os
import sys
import importlib.util

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_mod(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_mod("sp_instance_bf", os.path.join(_parent_dir, "instance.py"))
ShortestPathInstance = _inst.ShortestPathInstance
ShortestPathSolution = _inst.ShortestPathSolution


def bellman_ford(
    instance: ShortestPathInstance,
    source: int,
    target: int,
) -> ShortestPathSolution:
    """Find shortest path from source to target using Bellman-Ford.

    Args:
        instance: A ShortestPathInstance (may have negative weights).
        source: Source node.
        target: Target node.

    Returns:
        ShortestPathSolution.

    Raises:
        ValueError: If a negative-weight cycle is reachable from source.
    """
    n = instance.n
    dist = [float("inf")] * n
    prev = [-1] * n
    dist[source] = 0.0

    # Relax all edges V-1 times
    for _ in range(n - 1):
        updated = False
        for u, v, w in instance.edges:
            if dist[u] + w < dist[v] - 1e-10:
                dist[v] = dist[u] + w
                prev[v] = u
                updated = True
        if not updated:
            break

    # Check for negative cycles
    for u, v, w in instance.edges:
        if dist[u] + w < dist[v] - 1e-10:
            raise ValueError("Negative-weight cycle detected")

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


if __name__ == "__main__":
    from instance import negative_weight_graph

    inst = negative_weight_graph()
    sol = bellman_ford(inst, 0, 4)
    print(f"Bellman-Ford 0->4: {sol}")
