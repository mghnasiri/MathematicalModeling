"""
Kruskal's and Prim's Algorithms — Minimum Spanning Tree.

Problem: Minimum Spanning Tree (MST)
Complexity:
- Kruskal's: O(E log E) with union-find
- Prim's: O(E log V) with binary heap

Kruskal's: Sort edges by weight, greedily add if no cycle (union-find).
Prim's: Grow tree from node 0, always add cheapest crossing edge (heap).

Both produce optimal MSTs for connected graphs.

References:
    Kruskal, J.B. (1956). On the shortest spanning subtree of a graph
    and the traveling salesman problem. Proceedings of the American
    Mathematical Society, 7(1), 48-50.
    https://doi.org/10.1090/S0002-9939-1956-0078686-7

    Prim, R.C. (1957). Shortest connection networks and some
    generalizations. Bell System Technical Journal, 36(6), 1389-1401.
    https://doi.org/10.1002/j.1538-7305.1957.tb01515.x
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


_inst = _load_mod("mst_instance_ex", os.path.join(_parent_dir, "instance.py"))
MSTInstance = _inst.MSTInstance
MSTSolution = _inst.MSTSolution


class _UnionFind:
    """Disjoint set / union-find with path compression and union by rank."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True


def kruskal(instance: MSTInstance) -> MSTSolution:
    """Find MST using Kruskal's algorithm.

    Sort edges by weight, add greedily if no cycle.

    Args:
        instance: An MSTInstance.

    Returns:
        MSTSolution.
    """
    sorted_edges = sorted(instance.edges, key=lambda e: e[2])
    uf = _UnionFind(instance.n)
    tree_edges = []
    total = 0.0

    for u, v, w in sorted_edges:
        if uf.union(u, v):
            tree_edges.append((u, v, w))
            total += w
            if len(tree_edges) == instance.n - 1:
                break

    return MSTSolution(tree_edges=tree_edges, total_weight=total)


def prim(instance: MSTInstance) -> MSTSolution:
    """Find MST using Prim's algorithm.

    Grow tree from node 0, always add cheapest crossing edge.

    Args:
        instance: An MSTInstance.

    Returns:
        MSTSolution.
    """
    n = instance.n
    in_tree = [False] * n
    tree_edges = []
    total = 0.0

    # Priority queue: (weight, u, v) — edge from u to v
    pq: list[tuple[float, int, int]] = []
    in_tree[0] = True
    for v, w in instance.adjacency[0]:
        heapq.heappush(pq, (w, 0, v))

    while pq and len(tree_edges) < n - 1:
        w, u, v = heapq.heappop(pq)
        if in_tree[v]:
            continue
        in_tree[v] = True
        tree_edges.append((u, v, w))
        total += w

        for nv, nw in instance.adjacency[v]:
            if not in_tree[nv]:
                heapq.heappush(pq, (nw, v, nv))

    return MSTSolution(tree_edges=tree_edges, total_weight=total)


if __name__ == "__main__":
    from instance import simple_graph_6

    inst = simple_graph_6()
    k = kruskal(inst)
    p = prim(inst)
    print(f"Kruskal: {k}")
    print(f"Prim: {p}")
