"""
Hybrid Flow Shop (HFS) — Metaheuristics.

Problem notation: HFm | prmu | Cmax

Algorithms:
    - Simulated Annealing with swap/insertion neighborhood.

References:
    Ruiz, R. & Vázquez-Rodríguez, J.A. (2010). The hybrid flow shop
    scheduling problem. European Journal of Operational Research, 205(1),
    1-18. https://doi.org/10.1016/j.ejor.2009.09.024
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


_inst = _load_mod("hfs_instance_m", os.path.join(_this_dir, "instance.py"))
HFSInstance = _inst.HFSInstance
HFSSolution = _inst.HFSSolution

_heur = _load_mod("hfs_heuristics_m", os.path.join(_this_dir, "heuristics.py"))
neh_hfs = _heur.neh_hfs


def simulated_annealing(
    instance: HFSInstance,
    max_iterations: int = 50000,
    cooling_rate: float = 0.9995,
    seed: int | None = None,
    time_limit: float | None = None,
) -> HFSSolution:
    """Simulated Annealing for HFS.

    Args:
        instance: HFS instance.
        max_iterations: Maximum iterations.
        cooling_rate: Geometric cooling factor.
        seed: Random seed.
        time_limit: Wall-clock time limit in seconds.

    Returns:
        Best HFSSolution found.
    """
    rng = np.random.default_rng(seed)
    n = instance.n

    # Warm-start from NEH
    init = neh_hfs(instance)
    perm = list(init.permutation)
    cost = init.makespan

    best_perm = list(perm)
    best_cost = cost

    temp = best_cost * 0.1
    start = time.time()

    for it in range(max_iterations):
        if time_limit and time.time() - start > time_limit:
            break

        new_perm = list(perm)
        move = rng.integers(0, 2)

        if move == 0:
            # Swap
            i, j = rng.choice(n, 2, replace=False)
            new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
        else:
            # Insertion
            i = rng.integers(0, n)
            j = rng.integers(0, n - 1)
            job = new_perm.pop(i)
            new_perm.insert(j, job)

        new_cost = instance.makespan(new_perm)
        delta = new_cost - cost

        if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-15)):
            perm = new_perm
            cost = new_cost
            if cost < best_cost - 1e-10:
                best_cost = cost
                best_perm = list(perm)

        temp *= cooling_rate

    return HFSSolution(permutation=best_perm, makespan=best_cost)


if __name__ == "__main__":
    from instance import small_hfs_4x3

    inst = small_hfs_4x3()
    sol = simulated_annealing(inst, max_iterations=10000, seed=42)
    print(f"SA: {sol}")
    print(f"Permutation: {sol.permutation}")
