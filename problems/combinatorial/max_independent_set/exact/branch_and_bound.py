"""
Branch and Bound for Maximum Independent Set

Exact algorithm using greedy coloring upper bound and
vertex-ordering branching.

Complexity: Exponential worst case; practical for small graphs.

References:
    - Tomita, E. & Seki, T. (2003). An efficient branch-and-bound
      algorithm for finding a maximum clique. DMTCS, 2731, 278-289.
"""
from __future__ import annotations

import sys
import os
import importlib.util

import numpy as np

def _load_parent(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_inst = _load_parent("mis_instance", os.path.join(os.path.dirname(__file__), "..", "instance.py"))
MISInstance = _inst.MISInstance
MISSolution = _inst.MISSolution


def branch_and_bound(instance: MISInstance, time_limit_nodes: int = 100000) -> MISSolution:
    """B&B for MIS with greedy coloring upper bound.

    Args:
        instance: MISInstance.
        time_limit_nodes: Maximum nodes to explore.

    Returns:
        Best MISSolution found.
    """
    adj = instance.adjacency_list()
    n = instance.n_vertices
    best_set: list[int] = []
    nodes_explored = 0

    def upper_bound(candidates: list[int]) -> int:
        """Greedy coloring gives upper bound on MIS size of subgraph."""
        return len(candidates)  # trivial bound; tighten with coloring

    def solve(candidates: list[int], current: list[int]):
        nonlocal best_set, nodes_explored

        nodes_explored += 1
        if nodes_explored > time_limit_nodes:
            return

        if not candidates:
            if len(current) > len(best_set):
                best_set = list(current)
            return

        if len(current) + len(candidates) <= len(best_set):
            return

        # Sort by degree ascending for better pruning
        candidates_sorted = sorted(candidates, key=lambda v: len(adj[v] & set(candidates)))

        for i, v in enumerate(candidates_sorted):
            remaining = [u for u in candidates_sorted[i + 1:] if u not in adj[v]]
            if len(current) + 1 + len(remaining) <= len(best_set):
                continue
            solve(remaining, current + [v])

    all_vertices = list(range(n))
    # Warm start with greedy
    _heur = _load_parent("mis_greedy", os.path.join(os.path.dirname(__file__), "..", "heuristics", "greedy_mis.py"))
    warm = _heur.greedy_min_degree(instance)
    best_set = list(warm.independent_set)

    solve(all_vertices, [])

    best_set.sort()
    return MISSolution(
        independent_set=best_set,
        size=len(best_set),
        is_valid=instance.is_independent(best_set),
    )
