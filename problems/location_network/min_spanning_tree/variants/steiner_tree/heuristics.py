"""
Steiner Tree — Heuristics.

Algorithms:
    - Kou-Markowsky-Berman (KMB): Build MST on terminal-distance graph,
      map back to original graph. 2(1 - 1/l)-approximation where l = |S|.
    - Shortest Path Heuristic: iteratively connect nearest terminal.

Complexity: O(|S|^2 * n + n^3) for KMB (dominated by Floyd-Warshall).

References:
    Kou, L., Markowsky, G. & Berman, L. (1981). A fast algorithm for
    Steiner trees. Acta Informatica, 15(2), 141-145.
    https://doi.org/10.1007/BF00288961
"""

from __future__ import annotations

import sys
import os
import importlib.util

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


_inst = _load_mod("steiner_instance_h", os.path.join(_this_dir, "instance.py"))
SteinerTreeInstance = _inst.SteinerTreeInstance
SteinerTreeSolution = _inst.SteinerTreeSolution


def kmb_heuristic(instance: SteinerTreeInstance) -> SteinerTreeSolution:
    """Kou-Markowsky-Berman approximation algorithm.

    1. Compute all-pairs shortest paths.
    2. Build complete graph on terminals with shortest-path distances.
    3. Find MST of this terminal graph.
    4. Map MST edges back to shortest paths in original graph.
    5. Remove redundant edges (prune non-terminal leaves).

    Args:
        instance: SteinerTree instance.

    Returns:
        SteinerTreeSolution.
    """
    n = instance.n
    terminals = sorted(instance.terminals)
    if len(terminals) <= 1:
        return SteinerTreeSolution(tree_edges=[], total_weight=0.0)

    # All-pairs shortest paths with predecessor tracking
    dist = instance.adjacency_matrix()
    pred = np.full((n, n), -1, dtype=int)
    for u in range(n):
        for v in range(n):
            if dist[u][v] < np.inf and u != v:
                pred[u][v] = u

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    pred[i][j] = pred[k][j]

    # Build MST on terminal graph using Prim's
    nt = len(terminals)
    in_mst = [False] * nt
    in_mst[0] = True
    mst_edges = []

    for _ in range(nt - 1):
        best_w = np.inf
        best_i, best_j = -1, -1
        for i in range(nt):
            if not in_mst[i]:
                continue
            for j in range(nt):
                if in_mst[j]:
                    continue
                d = dist[terminals[i]][terminals[j]]
                if d < best_w:
                    best_w = d
                    best_i, best_j = i, j
        if best_j >= 0:
            in_mst[best_j] = True
            mst_edges.append((terminals[best_i], terminals[best_j]))

    # Map MST edges to original graph paths
    edge_set: set[tuple[int, int]] = set()
    edge_weights: dict[tuple[int, int], float] = {}
    adj = instance.adjacency_matrix()

    for s, t in mst_edges:
        # Trace path from s to t
        v = t
        while v != s:
            u = pred[s][v]
            if u < 0:
                break
            key = (min(u, v), max(u, v))
            if key not in edge_set:
                edge_set.add(key)
                edge_weights[key] = adj[u][v]
            v = u

    # Prune non-terminal leaves
    changed = True
    while changed:
        changed = False
        # Build degree map
        degree: dict[int, int] = {}
        for u, v in edge_set:
            degree[u] = degree.get(u, 0) + 1
            degree[v] = degree.get(v, 0) + 1
        to_remove = []
        for u, v in edge_set:
            if degree.get(u, 0) == 1 and u not in instance.terminals:
                to_remove.append((u, v))
            elif degree.get(v, 0) == 1 and v not in instance.terminals:
                to_remove.append((u, v))
        for e in to_remove:
            edge_set.discard(e)
            changed = True

    tree_edges = [(u, v, edge_weights[(u, v)]) for u, v in edge_set]
    total = sum(w for _, _, w in tree_edges)
    return SteinerTreeSolution(tree_edges=tree_edges, total_weight=total)


def shortest_path_heuristic(instance: SteinerTreeInstance) -> SteinerTreeSolution:
    """Iteratively connect nearest unconnected terminal via shortest path.

    Args:
        instance: SteinerTree instance.

    Returns:
        SteinerTreeSolution.
    """
    n = instance.n
    terminals = sorted(instance.terminals)
    if len(terminals) <= 1:
        return SteinerTreeSolution(tree_edges=[], total_weight=0.0)

    dist = instance.adjacency_matrix()
    pred = np.full((n, n), -1, dtype=int)
    for u in range(n):
        for v in range(n):
            if dist[u][v] < np.inf and u != v:
                pred[u][v] = u

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    pred[i][j] = pred[k][j]

    adj = instance.adjacency_matrix()
    connected = {terminals[0]}
    edge_set: set[tuple[int, int]] = set()
    edge_weights: dict[tuple[int, int], float] = {}

    remaining = set(terminals[1:])
    while remaining:
        best_d = np.inf
        best_src, best_tgt = -1, -1
        for s in connected:
            for t in remaining:
                if dist[s][t] < best_d:
                    best_d = dist[s][t]
                    best_src, best_tgt = s, t
        if best_tgt < 0:
            break

        # Add path from best_src to best_tgt
        v = best_tgt
        while v != best_src:
            u = pred[best_src][v]
            if u < 0:
                break
            key = (min(u, v), max(u, v))
            if key not in edge_set:
                edge_set.add(key)
                edge_weights[key] = adj[u][v]
            connected.add(v)
            v = u
        connected.add(best_tgt)
        remaining.discard(best_tgt)

    tree_edges = [(u, v, edge_weights[(u, v)]) for u, v in edge_set]
    total = sum(w for _, _, w in tree_edges)
    return SteinerTreeSolution(tree_edges=tree_edges, total_weight=total)


if __name__ == "__main__":
    from instance import small_steiner_6

    inst = small_steiner_6()
    sol1 = kmb_heuristic(inst)
    print(f"KMB: {sol1}")
    sol2 = shortest_path_heuristic(inst)
    print(f"SP heuristic: {sol2}")
