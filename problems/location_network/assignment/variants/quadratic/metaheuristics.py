"""
Quadratic Assignment Problem (QAP) — Metaheuristics.

Problem notation: QAP

Algorithms:
    - Simulated Annealing with pairwise swap neighborhood.

Complexity: O(max_iterations × n) per run.

References:
    Burkard, R.E. & Rendl, F. (1984). A thermodynamically motivated
    simulation procedure for combinatorial optimization problems.
    European Journal of Operational Research, 17(2), 169-174.
    https://doi.org/10.1016/0377-2217(84)90231-5
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


_inst = _load_mod("qap_instance_m", os.path.join(_this_dir, "instance.py"))
QAPInstance = _inst.QAPInstance
QAPSolution = _inst.QAPSolution

_heur = _load_mod("qap_heuristics_m", os.path.join(_this_dir, "heuristics.py"))
greedy_construction = _heur.greedy_construction
_swap_delta = _heur._swap_delta


def simulated_annealing(
    instance: QAPInstance,
    max_iterations: int = 50000,
    initial_temp: float = 0.0,
    cooling_rate: float = 0.9995,
    seed: int | None = None,
    time_limit: float | None = None,
) -> QAPSolution:
    """Simulated Annealing for QAP.

    Args:
        instance: QAP instance.
        max_iterations: Maximum number of iterations.
        initial_temp: Starting temperature (auto-calibrated if 0).
        cooling_rate: Geometric cooling factor.
        seed: Random seed for reproducibility.
        time_limit: Wall-clock time limit in seconds.

    Returns:
        Best QAPSolution found.
    """
    rng = np.random.default_rng(seed)
    n = instance.n

    # Warm-start from greedy
    sol = greedy_construction(instance)
    perm = list(sol.assignment)
    cost = sol.cost

    best_perm = list(perm)
    best_cost = cost

    # Auto-calibrate temperature
    if initial_temp <= 0:
        deltas = []
        for _ in range(min(200, n * n)):
            i, j = rng.choice(n, 2, replace=False)
            d = abs(_swap_delta(instance, perm, i, j))
            if d > 0:
                deltas.append(d)
        avg_delta = np.mean(deltas) if deltas else 1.0
        initial_temp = -avg_delta / math.log(0.5)

    temp = initial_temp
    start = time.time()

    for it in range(max_iterations):
        if time_limit and time.time() - start > time_limit:
            break

        i, j = rng.choice(n, 2, replace=False)
        delta = _swap_delta(instance, perm, i, j)

        if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-15)):
            perm[i], perm[j] = perm[j], perm[i]
            cost += delta

            if cost < best_cost - 1e-10:
                best_cost = cost
                best_perm = list(perm)

        temp *= cooling_rate

    return QAPSolution(assignment=best_perm, cost=best_cost)


if __name__ == "__main__":
    from instance import small_qap_4

    inst = small_qap_4()
    sol = simulated_annealing(inst, max_iterations=10000, seed=42)
    print(f"SA: {sol}")
    print(f"Assignment: {sol.assignment}")
