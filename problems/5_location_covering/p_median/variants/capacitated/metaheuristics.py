"""
Capacitated p-Median — Metaheuristics.

Algorithms:
    - Simulated Annealing with facility swap and customer reassignment.

References:
    Lorena, L.A.N. & Senne, E.L.F. (2004). A column generation approach
    to capacitated p-median problems. Computers & Operations Research,
    31(6), 863-876. https://doi.org/10.1016/S0305-0548(03)00039-X
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


_inst = _load_mod("cpmp_instance_m", os.path.join(_this_dir, "instance.py"))
CPMedianInstance = _inst.CPMedianInstance
CPMedianSolution = _inst.CPMedianSolution

_heur = _load_mod("cpmp_heuristics_m", os.path.join(_this_dir, "heuristics.py"))
greedy_add = _heur.greedy_add
_assign_customers = _heur._assign_customers


def simulated_annealing(
    instance: CPMedianInstance,
    max_iterations: int = 50000,
    cooling_rate: float = 0.9995,
    seed: int | None = None,
    time_limit: float | None = None,
) -> CPMedianSolution:
    """SA for capacitated p-Median.

    Swap open/closed facilities with capacity-aware customer reassignment.

    Args:
        instance: CPMedian instance.
        max_iterations: Maximum iterations.
        cooling_rate: Temperature decay factor.
        seed: Random seed.
        time_limit: Time limit in seconds.

    Returns:
        CPMedianSolution.
    """
    rng = np.random.default_rng(seed)

    init = greedy_add(instance)
    open_facs = list(init.open_facilities)
    _, cost = _assign_customers(instance, open_facs)

    best_facs = list(open_facs)
    best_cost = cost

    temp = best_cost * 0.1
    start = time.time()

    for it in range(max_iterations):
        if time_limit and time.time() - start > time_limit:
            break

        new_facs = list(open_facs)
        open_set = set(new_facs)
        closed = [i for i in range(instance.m) if i not in open_set]

        if not closed:
            break

        # Swap: close one, open one
        i_out = int(rng.integers(0, len(new_facs)))
        i_in = closed[int(rng.integers(0, len(closed)))]
        new_facs[i_out] = i_in

        assign, new_cost = _assign_customers(instance, new_facs)

        # Check capacity feasibility
        loads = np.zeros(instance.m)
        for j in range(instance.n):
            loads[assign[j]] += instance.demands[j]
        feasible = all(loads[f] <= instance.capacities[f] + 1e-6 for f in new_facs)

        if not feasible:
            temp *= cooling_rate
            continue

        delta = new_cost - cost
        if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-15)):
            open_facs = new_facs
            cost = new_cost
            if cost < best_cost:
                best_cost = cost
                best_facs = list(open_facs)

        temp *= cooling_rate

    assignments, total_cost = _assign_customers(instance, best_facs)
    return CPMedianSolution(open_facilities=best_facs,
                            assignments=assignments, total_cost=total_cost)


if __name__ == "__main__":
    from instance import small_cpmp_6

    inst = small_cpmp_6()
    sol = simulated_annealing(inst, max_iterations=10000, seed=42)
    print(f"SA: {sol}")
