"""
Greedy Heuristics for Maximum Independent Set

1. Greedy minimum degree: iteratively add vertex with fewest neighbors
   among remaining vertices, remove it and its neighbors.
2. Greedy random: randomized greedy for diversity.

Complexity: O(V + E) per pass.

References:
    - Halldórsson, M.M. & Radhakrishnan, J. (1997). Greed is good:
      Approximating independent sets in sparse and bounded-degree graphs.
      Algorithmica, 18(1), 145-163.
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


def greedy_min_degree(instance: MISInstance) -> MISSolution:
    """Add vertex with minimum degree, remove it and neighbors."""
    adj = instance.adjacency_list()
    remaining = set(range(instance.n_vertices))
    independent = []

    while remaining:
        # Pick vertex with minimum degree among remaining
        v = min(remaining, key=lambda u: len(adj[u] & remaining))
        independent.append(v)
        # Remove v and its neighbors
        to_remove = (adj[v] & remaining) | {v}
        remaining -= to_remove

    independent.sort()
    return MISSolution(
        independent_set=independent,
        size=len(independent),
        is_valid=instance.is_independent(independent),
    )


def greedy_random(instance: MISInstance, n_starts: int = 10,
                  seed: int = 42) -> MISSolution:
    """Multi-start random greedy."""
    rng = np.random.default_rng(seed)
    adj = instance.adjacency_list()
    best = MISSolution([], 0, True)

    for _ in range(n_starts):
        remaining = set(range(instance.n_vertices))
        independent = []
        order = list(rng.permutation(instance.n_vertices))

        for v in order:
            if v not in remaining:
                continue
            independent.append(v)
            remaining -= (adj[v] & remaining) | {v}

        if len(independent) > best.size:
            independent.sort()
            best = MISSolution(independent, len(independent),
                               instance.is_independent(independent))

    return best
