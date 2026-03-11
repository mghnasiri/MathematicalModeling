"""
Simulated Annealing for Bounded Knapsack.

Problem: BKP

Integer-vector encoding: x[i] ∈ {0,...,b_i}. Neighborhood: increment
or decrement a random item's quantity.

Warm-started with greedy heuristic.

References:
    Kellerer, H., Pferschy, U. & Pisinger, D. (2004). Knapsack Problems.
"""

from __future__ import annotations

import sys
import os
import math
import time
import importlib.util

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


_inst = _load_mod("bkp_instance_meta", os.path.join(_this_dir, "instance.py"))
BKPInstance = _inst.BKPInstance
BKPSolution = _inst.BKPSolution

_heur = _load_mod("bkp_heur_meta", os.path.join(_this_dir, "heuristics.py"))
greedy_density = _heur.greedy_density


def simulated_annealing(
    instance: BKPInstance,
    max_iterations: int = 20000,
    initial_temp: float | None = None,
    cooling_rate: float = 0.9995,
    time_limit: float | None = None,
    seed: int | None = None,
) -> BKPSolution:
    rng = np.random.default_rng(seed)
    n = instance.n
    start_time = time.time()

    init_sol = greedy_density(instance)
    qty = init_sol.quantities[:]
    current_val = init_sol.value
    current_weight = sum(instance.weights[i] * qty[i] for i in range(n))

    best_qty = qty[:]
    best_val = current_val

    if initial_temp is None:
        initial_temp = max(best_val * 0.1, 10.0)
    temp = initial_temp

    for iteration in range(max_iterations):
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        i = rng.integers(0, n)
        delta_qty = rng.choice([-1, 1])
        new_q = qty[i] + delta_qty

        if new_q < 0 or new_q > instance.bounds[i]:
            temp *= cooling_rate
            continue

        new_weight = current_weight + delta_qty * instance.weights[i]
        if new_weight > instance.capacity + 1e-10:
            temp *= cooling_rate
            continue

        new_val = current_val + delta_qty * instance.values[i]
        delta = -(new_val - current_val)  # minimize negative value

        if delta < 0 or (temp > 0 and rng.random() < math.exp(-delta / max(temp, 1e-10))):
            qty[i] = new_q
            current_val = new_val
            current_weight = new_weight

            if current_val > best_val + 1e-10:
                best_val = current_val
                best_qty = qty[:]

        temp *= cooling_rate

    return BKPSolution(quantities=best_qty, value=best_val)


if __name__ == "__main__":
    inst = BKPInstance.random(n=10, seed=42)
    gr = greedy_density(inst)
    print(f"Greedy: {gr.value:.0f}")
    sa = simulated_annealing(inst, seed=42)
    print(f"SA: {sa.value:.0f}")
