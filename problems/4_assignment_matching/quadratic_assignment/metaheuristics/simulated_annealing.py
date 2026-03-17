"""
Simulated Annealing for QAP

Pairwise swap neighborhood with fast delta evaluation.

Complexity: O(max_iter * n) per run using delta_swap.

References:
    - Burkard, R.E. & Rendl, F. (1984). A thermodynamically motivated
      simulation procedure for combinatorial optimization problems.
      EJOR, 17(2), 169-174.
"""
from __future__ import annotations

import sys
import os
import importlib.util
import math

import numpy as np

def _load_parent(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_base = os.path.dirname(os.path.dirname(__file__))
_inst = _load_parent("qap_instance", os.path.join(_base, "instance.py"))
_heur = _load_parent("qap_greedy", os.path.join(_base, "heuristics", "greedy_qap.py"))

QAPInstance = _inst.QAPInstance
QAPSolution = _inst.QAPSolution


def simulated_annealing(
    instance: QAPInstance,
    max_iterations: int = 10000,
    initial_temp: float = 100.0,
    cooling_rate: float = 0.9995,
    seed: int = 42,
) -> QAPSolution:
    """SA for QAP with pairwise swap."""
    rng = np.random.default_rng(seed)
    init = _heur.greedy_construction(instance)
    current = list(init.permutation)
    current_obj = init.objective
    best = list(current)
    best_obj = current_obj
    temp = initial_temp

    for _ in range(max_iterations):
        r = rng.integers(instance.n)
        s = rng.integers(instance.n)
        while s == r:
            s = rng.integers(instance.n)

        delta = instance.delta_swap(current, r, s)

        if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-10)):
            current[r], current[s] = current[s], current[r]
            current_obj += delta
            if current_obj < best_obj:
                best = list(current)
                best_obj = current_obj

        temp *= cooling_rate

    return QAPSolution(permutation=best, objective=best_obj)
