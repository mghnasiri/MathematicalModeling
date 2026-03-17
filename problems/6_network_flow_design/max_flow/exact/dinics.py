"""
Dinic's Algorithm for Maximum Flow.

Problem: Max-Flow

Dinic's algorithm (1970) finds maximum flow by repeatedly constructing
a layered graph (BFS from source to sink) and then pushing flow along
blocking flows (DFS) in the layered graph. It is more efficient than
Edmonds-Karp for dense graphs.

Algorithm:
    1. Build layered graph using BFS from source.
    2. If sink is unreachable, stop (max flow found).
    3. Find blocking flow in layered graph using DFS.
    4. Augment flow and update residual graph.
    5. Repeat from step 1.

Complexity: O(V^2 * E) — an improvement over Edmonds-Karp's O(V * E^2)
            for dense graphs. For unit-capacity graphs, O(E * sqrt(V)).

Reference:
    Dinic, E.A. (1970). Algorithm for solution of a problem of maximum
    flow in a network with power estimation. Soviet Mathematics Doklady,
    11, 1277-1280.

    Even, S. & Tarjan, R.E. (1975). Network flow and testing graph
    connectivity. SIAM Journal on Computing, 4(4), 507-518.
    https://doi.org/10.1137/0204043
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


_inst = _load_mod("mf_instance_dinic", os.path.join(_parent_dir, "instance.py"))
MaxFlowInstance = _inst.MaxFlowInstance
MaxFlowSolution = _inst.MaxFlowSolution


def _bfs_level(
    residual: np.ndarray,
    source: int,
    sink: int,
    n: int,
) -> list[int] | None:
    """Build level graph using BFS. Returns level array or None if sink unreachable."""
    level = [-1] * n
    level[source] = 0
    queue = deque([source])

    while queue:
        u = queue.popleft()
        for v in range(n):
            if level[v] == -1 and residual[u][v] > 1e-10:
                level[v] = level[u] + 1
                queue.append(v)

    return level if level[sink] != -1 else None


def _dfs_blocking(
    residual: np.ndarray,
    level: list[int],
    u: int,
    sink: int,
    pushed: float,
    n: int,
    iter_ptr: list[int],
) -> float:
    """Send blocking flow via DFS in level graph.

    Uses iter_ptr to avoid re-scanning dead-end edges (optimization).
    """
    if u == sink:
        return pushed

    while iter_ptr[u] < n:
        v = iter_ptr[u]
        if level[v] == level[u] + 1 and residual[u][v] > 1e-10:
            d = _dfs_blocking(
                residual, level, v, sink,
                min(pushed, residual[u][v]), n, iter_ptr,
            )
            if d > 1e-10:
                residual[u][v] -= d
                residual[v][u] += d
                return d
        iter_ptr[u] += 1

    return 0.0


def dinics(instance: MaxFlowInstance) -> MaxFlowSolution:
    """Solve max-flow using Dinic's algorithm.

    Args:
        instance: MaxFlowInstance.

    Returns:
        MaxFlowSolution with max flow, flow matrix, and min-cut.
    """
    n = instance.n
    source = instance.source
    sink = instance.sink

    residual = instance.capacity_matrix.astype(float).copy()
    total_flow = 0.0

    while True:
        level = _bfs_level(residual, source, sink, n)
        if level is None:
            break

        while True:
            iter_ptr = [0] * n
            f = _dfs_blocking(
                residual, level, source, sink, float("inf"), n, iter_ptr,
            )
            if f < 1e-10:
                break
            total_flow += f

    # Compute flow matrix
    flow = instance.capacity_matrix.astype(float).copy() - residual
    # Only positive flows
    flow = np.maximum(flow, 0.0)

    # Extract min-cut via BFS on residual graph
    visited = set()
    queue = deque([source])
    visited.add(source)
    while queue:
        u = queue.popleft()
        for v in range(n):
            if v not in visited and residual[u][v] > 1e-10:
                visited.add(v)
                queue.append(v)

    s_side = sorted(visited)
    t_side = sorted(set(range(n)) - visited)

    return MaxFlowSolution(
        max_flow=total_flow,
        flow_matrix=flow,
        min_cut=(s_side, t_side),
    )


if __name__ == "__main__":
    from instance import simple_flow_4, two_path_flow

    print("=== Dinic's on simple4 (max flow = 26) ===")
    inst = simple_flow_4()
    sol = dinics(inst)
    print(f"Max flow: {sol.max_flow}")
    print(f"Min cut: {sol.min_cut}")

    print("\n=== Dinic's on two_path5 (max flow = 5) ===")
    inst = two_path_flow()
    sol = dinics(inst)
    print(f"Max flow: {sol.max_flow}")
    print(f"Min cut: {sol.min_cut}")
