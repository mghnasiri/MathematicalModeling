"""
Maximum Weight Matching — Heuristics.

Algorithms:
    - Greedy max-weight: greedily select highest-weight edges.
    - Augmenting path (exact): adapted Hungarian for rectangular matrices.

References:
    Kuhn, H.W. (1955). The Hungarian method for the assignment problem.
    Naval Research Logistics Quarterly, 2(1-2), 83-97.
    https://doi.org/10.1002/nav.3800020109
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


_inst = _load_mod("maxmatch_instance_h", os.path.join(_this_dir, "instance.py"))
MaxMatchingInstance = _inst.MaxMatchingInstance
MaxMatchingSolution = _inst.MaxMatchingSolution


def greedy_matching(instance: MaxMatchingInstance) -> MaxMatchingSolution:
    """Greedy: select edges in decreasing weight order, skip conflicts.

    Args:
        instance: MaxMatchingInstance.

    Returns:
        MaxMatchingSolution.
    """
    nw, nt = instance.n_workers, instance.n_tasks
    # Collect all edges with positive weight
    edges = []
    for w in range(nw):
        for t in range(nt):
            if instance.weights[w][t] > 0:
                edges.append((instance.weights[w][t], w, t))
    edges.sort(reverse=True)

    matched_w: set[int] = set()
    matched_t: set[int] = set()
    matching: list[tuple[int, int]] = []

    for wt, w, t in edges:
        if w not in matched_w and t not in matched_t:
            matching.append((w, t))
            matched_w.add(w)
            matched_t.add(t)

    total = instance.matching_weight(matching)
    return MaxMatchingSolution(matching=matching, total_weight=total)


def hungarian_max(instance: MaxMatchingInstance) -> MaxMatchingSolution:
    """Hungarian method adapted for maximum weight matching on rectangular matrices.

    Converts to a square minimization problem and solves optimally.

    Args:
        instance: MaxMatchingInstance.

    Returns:
        Optimal MaxMatchingSolution.
    """
    nw, nt = instance.n_workers, instance.n_tasks
    n = max(nw, nt)

    # Pad to square, use negative weights for minimization
    cost = np.zeros((n, n))
    max_w = instance.weights.max() if instance.weights.size > 0 else 0
    cost[:nw, :nt] = max_w - instance.weights

    # Hungarian algorithm (Kuhn-Munkres)
    u = np.zeros(n + 1)
    v = np.zeros(n + 1)
    p = np.zeros(n + 1, dtype=int)
    way = np.zeros(n + 1, dtype=int)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, float("inf"))
        used = np.zeros(n + 1, dtype=bool)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0

            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j

            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1
            if p[j0] == 0:
                break

        while j0:
            p[j0] = p[way[j0]]
            j0 = way[j0]

    # Extract matching
    matching = []
    for j in range(1, n + 1):
        if p[j] > 0:
            w = p[j] - 1
            t = j - 1
            if w < nw and t < nt and instance.weights[w][t] > 0:
                matching.append((w, t))

    total = instance.matching_weight(matching)
    return MaxMatchingSolution(matching=matching, total_weight=total)


if __name__ == "__main__":
    from instance import small_matching_4x5

    inst = small_matching_4x5()
    gr = greedy_matching(inst)
    print(f"Greedy: {gr}")
    hu = hungarian_max(inst)
    print(f"Hungarian: {hu}")
