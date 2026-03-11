"""
Lot Streaming Flow Shop — Metaheuristics.

Algorithms:
    - Simulated Annealing with swap/insertion.

References:
    Potts, C.N. & Baker, K.R. (2000). Flow shop scheduling with lot
    streaming. Operations Research Letters, 26(6), 297-303.
    https://doi.org/10.1016/S0167-6377(00)00011-0
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


_inst = _load_mod("lotstream_instance_m", os.path.join(_this_dir, "instance.py"))
LotStreamInstance = _inst.LotStreamInstance
LotStreamSolution = _inst.LotStreamSolution

_heur = _load_mod("lotstream_heuristics_m", os.path.join(_this_dir, "heuristics.py"))
neh_ls = _heur.neh_ls


def simulated_annealing(
    instance: LotStreamInstance,
    max_iterations: int = 50000,
    cooling_rate: float = 0.9995,
    seed: int | None = None,
    time_limit: float | None = None,
) -> LotStreamSolution:
    """SA for Lot Streaming Flow Shop.

    Args:
        instance: LotStreamInstance.
        max_iterations: Maximum iterations.
        cooling_rate: Geometric cooling factor.
        seed: Random seed.
        time_limit: Wall-clock time limit in seconds.

    Returns:
        Best LotStreamSolution found.
    """
    rng = np.random.default_rng(seed)
    n = instance.n

    init = neh_ls(instance)
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
        if rng.integers(0, 2) == 0:
            i, j = rng.choice(n, 2, replace=False)
            new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
        else:
            i = rng.integers(0, n)
            j = rng.integers(0, n - 1)
            job = new_perm.pop(i)
            new_perm.insert(j, job)

        new_cost = instance.makespan_streaming(new_perm)
        delta = new_cost - cost

        if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-15)):
            perm = new_perm
            cost = new_cost
            if cost < best_cost - 1e-10:
                best_cost = cost
                best_perm = list(perm)

        temp *= cooling_rate

    ms_ns = instance.makespan_no_streaming(best_perm)
    return LotStreamSolution(permutation=best_perm, makespan=best_cost,
                             makespan_no_stream=ms_ns)


if __name__ == "__main__":
    from instance import small_ls_4x3

    inst = small_ls_4x3()
    sol = simulated_annealing(inst, max_iterations=10000, seed=42)
    print(f"SA: {sol}")
