"""
Subset Sum Problem — Metaheuristics.

Algorithms:
    - Simulated Annealing with add/remove/swap neighborhood.

References:
    Kellerer, H., Pferschy, U. & Pisinger, D. (2004). Knapsack Problems.
    Springer. https://doi.org/10.1007/978-3-540-24777-7
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


_inst = _load_mod("ssp_instance_m", os.path.join(_this_dir, "instance.py"))
SubsetSumInstance = _inst.SubsetSumInstance
SubsetSumSolution = _inst.SubsetSumSolution

_heur = _load_mod("ssp_heuristics_m", os.path.join(_this_dir, "heuristics.py"))
greedy_largest = _heur.greedy_largest


def simulated_annealing(
    instance: SubsetSumInstance,
    max_iterations: int = 50000,
    cooling_rate: float = 0.9995,
    seed: int | None = None,
    time_limit: float | None = None,
) -> SubsetSumSolution:
    """Simulated Annealing for Subset Sum.

    Objective: maximize total subject to total <= target.

    Args:
        instance: Subset Sum instance.
        max_iterations: Maximum iterations.
        cooling_rate: Geometric cooling factor.
        seed: Random seed.
        time_limit: Wall-clock time limit in seconds.

    Returns:
        Best SubsetSumSolution found.
    """
    rng = np.random.default_rng(seed)
    n = instance.n
    T = instance.target
    values = instance.values

    init = greedy_largest(instance)
    in_set = np.zeros(n, dtype=bool)
    for i in init.selected:
        in_set[i] = True
    total = init.total

    best_set = in_set.copy()
    best_total = total

    temp = T * 0.1
    start = time.time()

    for it in range(max_iterations):
        if time_limit and time.time() - start > time_limit:
            break

        move = rng.integers(0, 3)
        new_set = in_set.copy()
        new_total = total

        if move == 0:
            # Add a random element not in set
            out = np.where(~in_set)[0]
            if len(out) > 0:
                i = rng.choice(out)
                v = int(values[i])
                if new_total + v <= T:
                    new_set[i] = True
                    new_total += v

        elif move == 1:
            # Remove a random element from set
            ins = np.where(in_set)[0]
            if len(ins) > 0:
                i = rng.choice(ins)
                new_set[i] = False
                new_total -= int(values[i])

        elif move == 2:
            # Swap: remove one, add another
            ins = np.where(in_set)[0]
            out = np.where(~in_set)[0]
            if len(ins) > 0 and len(out) > 0:
                i_out = rng.choice(ins)
                i_in = rng.choice(out)
                new_total_candidate = new_total - int(values[i_out]) + int(values[i_in])
                if new_total_candidate <= T:
                    new_set[i_out] = False
                    new_set[i_in] = True
                    new_total = new_total_candidate

        # Objective: maximize total (minimize -total)
        delta = -(new_total - total)  # negative delta means improvement
        if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-15)):
            in_set = new_set
            total = new_total
            if total > best_total:
                best_total = total
                best_set = in_set.copy()

        temp *= cooling_rate

    selected = [int(i) for i in np.where(best_set)[0]]
    return SubsetSumSolution(selected=selected, total=best_total)


if __name__ == "__main__":
    from instance import small_ssp_6

    inst = small_ssp_6()
    sol = simulated_annealing(inst, max_iterations=10000, seed=42)
    print(f"SA: {sol}")
