"""
Constructive Heuristics for QAP

1. Random construction: random permutation (baseline).
2. Greedy construction: assign facility-location pairs that minimize
   incremental cost.

Complexity: O(n^3) for greedy.

References:
    - Burkard, R.E., Dell'Amico, M. & Martello, S. (2012). Assignment
      Problems. SIAM, Chapter 12.
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

_inst = _load_parent("qap_instance", os.path.join(os.path.dirname(__file__), "..", "instance.py"))
QAPInstance = _inst.QAPInstance
QAPSolution = _inst.QAPSolution


def greedy_construction(instance: QAPInstance) -> QAPSolution:
    """Greedy: assign pairs to minimize incremental cost."""
    n = instance.n
    flow_sum = instance.flow.sum(axis=1) + instance.flow.sum(axis=0)
    dist_sum = instance.distance.sum(axis=1) + instance.distance.sum(axis=0)

    # Assign highest-flow facility to lowest-distance-sum location
    fac_order = np.argsort(-flow_sum)
    loc_order = np.argsort(dist_sum)

    perm = [-1] * n
    for k in range(n):
        perm[int(fac_order[k])] = int(loc_order[k])

    return QAPSolution(
        permutation=perm,
        objective=instance.objective(perm),
    )


def random_construction(instance: QAPInstance, seed: int = 42) -> QAPSolution:
    """Random permutation as baseline."""
    rng = np.random.default_rng(seed)
    perm = list(rng.permutation(instance.n))
    return QAPSolution(permutation=perm, objective=instance.objective(perm))
