"""
Edmonds-Karp Algorithm — Maximum flow via BFS augmenting paths.

Problem: Maximum Flow
Complexity: O(V * E^2)

The Edmonds-Karp algorithm is a specific implementation of the
Ford-Fulkerson method that uses BFS to find shortest augmenting
paths (fewest edges). This guarantees polynomial-time convergence
regardless of edge capacities.

Also computes the minimum cut (S, T) partition via reachability
in the residual graph after termination.

References:
    Edmonds, J. & Karp, R.M. (1972). Theoretical improvements in
    algorithmic efficiency for network flow problems. Journal of the
    ACM, 19(2), 248-264.
    https://doi.org/10.1145/321694.321699

    Ford, L.R. & Fulkerson, D.R. (1956). Maximal flow through a
    network. Canadian Journal of Mathematics, 8, 399-404.
    https://doi.org/10.4153/CJM-1956-045-5
"""

from __future__ import annotations

import os
import sys
import importlib.util
from collections import deque

import numpy as np

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_mod(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_inst = _load_mod("mf_instance_ek", os.path.join(_parent_dir, "instance.py"))
MaxFlowInstance = _inst.MaxFlowInstance
MaxFlowSolution = _inst.MaxFlowSolution


def _bfs(
    residual: np.ndarray, source: int, sink: int, parent: list[int]
) -> bool:
    """BFS to find augmenting path in residual graph.

    Args:
        residual: Residual capacity matrix.
        source: Source node.
        sink: Sink node.
        parent: Parent array (modified in-place).

    Returns:
        True if augmenting path exists.
    """
    n = residual.shape[0]
    visited = [False] * n
    visited[source] = True
    queue = deque([source])

    while queue:
        u = queue.popleft()
        for v in range(n):
            if not visited[v] and residual[u][v] > 1e-10:
                visited[v] = True
                parent[v] = u
                if v == sink:
                    return True
                queue.append(v)

    return False


def edmonds_karp(instance: MaxFlowInstance) -> MaxFlowSolution:
    """Compute maximum flow using Edmonds-Karp (BFS Ford-Fulkerson).

    Args:
        instance: A MaxFlowInstance.

    Returns:
        MaxFlowSolution with max flow, flow matrix, and min cut.
    """
    n = instance.n
    source, sink = instance.source, instance.sink
    residual = instance.capacity_matrix.copy()
    flow_matrix = np.zeros((n, n))

    max_flow = 0.0
    parent = [-1] * n

    while _bfs(residual, source, sink, parent):
        # Find bottleneck
        bottleneck = float("inf")
        v = sink
        while v != source:
            u = parent[v]
            bottleneck = min(bottleneck, residual[u][v])
            v = u

        # Update residual capacities and flow
        v = sink
        while v != source:
            u = parent[v]
            residual[u][v] -= bottleneck
            residual[v][u] += bottleneck
            flow_matrix[u][v] += bottleneck
            flow_matrix[v][u] -= bottleneck
            v = u

        max_flow += bottleneck
        parent = [-1] * n

    # Clean up reverse flows
    for u in range(n):
        for v in range(n):
            if flow_matrix[u][v] < 0:
                flow_matrix[u][v] = 0.0

    # Find min cut via BFS reachability in residual graph
    visited = [False] * n
    visited[source] = True
    queue = deque([source])
    while queue:
        u = queue.popleft()
        for v in range(n):
            if not visited[v] and residual[u][v] > 1e-10:
                visited[v] = True
                queue.append(v)

    s_set = [i for i in range(n) if visited[i]]
    t_set = [i for i in range(n) if not visited[i]]

    return MaxFlowSolution(
        max_flow=max_flow,
        flow_matrix=flow_matrix,
        min_cut=(s_set, t_set),
    )


if __name__ == "__main__":
    from instance import simple_flow_4, two_path_flow

    for name, fn in [("simple4", simple_flow_4), ("two_path", two_path_flow)]:
        inst = fn()
        sol = edmonds_karp(inst)
        print(f"{name}: max_flow={sol.max_flow}, min_cut={sol.min_cut}")
