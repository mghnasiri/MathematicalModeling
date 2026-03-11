"""
Tardiness Flow Shop — Metaheuristics.

Algorithms:
    - SA with swap/insertion.

References:
    Vallada, E. & Ruiz, R. (2010). Genetic algorithms with path
    relinking for the minimum tardiness permutation flowshop problem.
    Omega, 38(1-2), 57-67.
    https://doi.org/10.1016/j.omega.2009.04.002
"""

from __future__ import annotations

import math
import sys
import os
import importlib.util
import time

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


_inst = _load_mod("tfs_instance_m", os.path.join(_this_dir, "instance.py"))
TardinessFlowShopInstance = _inst.TardinessFlowShopInstance
TardinessFlowShopSolution = _inst.TardinessFlowShopSolution

_heur = _load_mod("tfs_heuristics_m", os.path.join(_this_dir, "heuristics.py"))
neh_tardiness = _heur.neh_tardiness


def simulated_annealing(
    instance: TardinessFlowShopInstance,
    max_iterations: int = 50000,
    cooling_rate: float = 0.9995,
    seed: int | None = None,
    time_limit: float | None = None,
) -> TardinessFlowShopSolution:
    """SA for Tardiness Flow Shop."""
    rng = np.random.default_rng(seed)
    n = instance.n

    init = neh_tardiness(instance)
    perm = list(init.permutation)
    cost = init.total_weighted_tardiness

    best_perm = list(perm)
    best_cost = cost

    temp = max(best_cost * 0.1, 1.0)
    start = time.time()

    for it in range(max_iterations):
        if time_limit and time.time() - start > time_limit:
            break

        new_perm = list(perm)
        if rng.integers(0, 2) == 0:
            i, j = rng.choice(n, 2, replace=False)
            new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
        else:
            i = rng.integers(0, n)
            j = rng.integers(0, n - 1)
            job = new_perm.pop(i)
            new_perm.insert(j, job)

        new_cost = instance.total_weighted_tardiness(new_perm)
        delta = new_cost - cost

        if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-15)):
            perm = new_perm
            cost = new_cost
            if cost < best_cost - 1e-10:
                best_cost = cost
                best_perm = list(perm)

        temp *= cooling_rate

    return TardinessFlowShopSolution(permutation=best_perm,
                                     total_weighted_tardiness=best_cost)


if __name__ == "__main__":
    from instance import small_tfs_4x3

    inst = small_tfs_4x3()
    sol = simulated_annealing(inst, max_iterations=10000, seed=42)
    print(f"SA: {sol}")
